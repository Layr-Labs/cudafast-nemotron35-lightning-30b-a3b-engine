//! The session backend: the real engine behind `cuda-engine` on the box.
//!
//! The engine is the pinned vLLM build (vendored at `vllm/`) running the pinned
//! Nemotron 3.5 Lightning checkpoint inside ONE resident process per benchmark
//! window (`harness/vllm_resident/`, booted by `tools/serve-up.sh`). The
//! per-phase `cuda-engine` process drives it through the [`EngineSession`]
//! seam over a Unix socket (`src/resident.rs`); nothing is loaded here.
//!
//! Verb translation, in the session's terms:
//!
//! * `prefill(prompt)` / `decode_begin(seed)` / `free_decode_begin(seed, ..)`
//!   -> `sync(prompt)`; the next token is the argmax of the logits the sync
//!   leaves behind. `prefill`, `free_decode_begin` and the free-run
//!   `correctness` reference read nothing but argmaxes, so they sync
//!   ARGMAX-ONLY ([`EngineSession::sync_argmax_only`]): the engine computes no
//!   top-k logits for any position of those sequences. Only the
//!   teacher-forced gates (`correctness_begin`, and `step` behind
//!   `correctness_step` / `decode_step`) ask for top-8 logits, and they sync
//!   with them;
//! * `step(input)` -> `eval(input)` then argmax + top-8 logits;
//!   `correctness_step` and `decode_step` share it;
//! * `free_decode_run(count)` -> on the SERIAL route, `count` plain evals; on
//!   the MTP route, speculative cycles (one `spec_run` request when the
//!   resident offers it) until `count` tokens are committed. Each cycle
//!   commits its input token plus every draft the target's own argmax
//!   confirmed, so the stream is the serial stream up to kernel numerics.
//!
//! The counters are the session's own speculation counters read before and
//! after the phase: `drafts` (draft tokens proposed into a verify), `hits`
//! (drafts the target accepted), `quenches`, and the diagnostic
//! `verify_replay_disagreements`. Nothing here computes a rate.
//!
//! DEPTH. The track envelope is 1..=6 draft tokens per cycle and the pinned
//! engine implements all six, so an explicit 1..=6 all run. A request outside
//! it is refused by name ([`EngineError::DepthOutOfEnvelope`]) rather than
//! clamped and echoed wrong.

use std::sync::{Arc, Mutex};

use crate::engine::{Engine, EngineError, EngineFactory, FreeRunResult, Route, Step};
use crate::protocol::{CorrectnessTraceLogit, EffectiveSpec, EffectiveSpecMTP, TOP_LOGITS_K};

/// The lowest depth the track envelope names (`permitted_draft_depths`).
pub const MTP_MIN_DEPTH: i64 = 1;
/// The highest depth the track envelope names.
pub const MTP_MAX_DEPTH: i64 = 6;
/// The depth the pinned engine implements: vLLM's `mtp` method runs the one
/// native MTP layer recursively for `num_speculative_tokens` drafts, so every
/// depth in the envelope is served. It MUST equal the resident's
/// `IMPLEMENTED_DEPTH` (harness/vllm_resident/resident.py); the test
/// `implemented_depth_agrees_with_the_resident` holds the two together.
pub const ENGINE_IMPLEMENTED_DEPTH: i64 = 6;

/// The speculative cycle's counters, as the engine reports them. Read before
/// and after a leg and take the difference; nothing here computes a rate.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SpecCounters {
    /// Rounds that carried a draft into a verify (the port's `drafted`).
    pub drafts: u64,
    /// Drafts the target's own argmax confirmed (the port's `accepted`).
    pub hits: u64,
    /// Adaptive-disable events. 0 on this path; the tripwire below stays armed.
    pub quenches: u64,
    /// DIAGNOSTIC (the port's `verify_replay_disagreements`): rejecting rounds
    /// where the batched verify's row-0 argmax and the one-row replay of the
    /// same position differed. The replay stands, so this is never a refusal
    /// and nothing here enforces a ceiling; benchd may seal one later.
    pub verify_replay_disagreements: u64,
}

/// One cycle of a batched speculative run ([`EngineSession::spec_run`]): the
/// tokens the cycle committed, the fed token first, and the frontier argmax it
/// left, which is the token the next cycle is fed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecRound {
    pub committed: Vec<i64>,
    pub frontier: i64,
}

/// The session seam. [`FfiSession`] is the real engine; tests inject a scripted
/// one, so the verb translation and the counter accounting are proven with no
/// GPU. It is the ONLY thing that touches the engine.
pub trait EngineSession: Send {
    /// Whether the engine was opened with the MTP drafter armed.
    fn mtp_armed(&self) -> bool;
    /// Synchronize the session to `tokens`; logits for the last position follow.
    fn sync(&mut self, tokens: &[i64]) -> Result<(), String>;
    /// [`EngineSession::sync`] for a caller that needs the frontier token and
    /// nothing else (a timed prefill): the engine may stop generating at the
    /// frontier, so no decode runs on after the phase has its answer, and it
    /// may skip the frontier's top-k logits, which nobody reads.
    fn sync_to_frontier(&mut self, tokens: &[i64]) -> Result<(), String> {
        self.sync_argmax_only(tokens)
    }
    /// [`EngineSession::sync`] for a caller that reads only argmaxes on this
    /// sequence (the free run, the free-run correctness reference): the engine
    /// may skip the top-k logits of every position on it, and
    /// [`EngineSession::top_logits`] is not asked of it. The tokens are the
    /// same: greedy selection never reads the top-k.
    fn sync_argmax_only(&mut self, tokens: &[i64]) -> Result<(), String> {
        self.sync(tokens)
    }
    /// Append and evaluate one token; logits for the new position follow.
    fn eval(&mut self, token: i64) -> Result<(), String>;
    /// The greedy token of the current logits.
    fn argmax(&mut self) -> i64;
    /// The top-k logits of the current position, highest first.
    fn top_logits(&mut self, k: usize) -> Vec<CorrectnessTraceLogit>;
    /// One speculative cycle on `first_token` with `budget` tokens still wanted.
    /// Returns the committed tokens (`[first_token]` or `[first_token, draft]`).
    fn eval_speculative(&mut self, first_token: i64, budget: i64) -> Result<Vec<i64>, String>;
    /// A whole speculative free run of `count` tokens in one engine call, when
    /// the session can run the cycles without a round trip per cycle: each
    /// cycle's committed tokens and the frontier argmax it left, in order. It
    /// must equal driving [`EngineSession::eval_speculative`] cycle by cycle, each
    /// with the budget still wanted. `None` means drive the cycles one by one.
    fn spec_run(
        &mut self,
        _first_token: i64,
        _count: i64,
    ) -> Option<Result<Vec<SpecRound>, String>> {
        None
    }
    /// The speculative cycle's counters. `&mut` because a session that reaches
    /// the engine over a socket has to send a request to read them.
    fn spec_counters(&mut self) -> SpecCounters;
    /// The model's end-of-sequence token id, when the engine knows one.
    fn eos_token(&self) -> Option<i64> {
        None
    }
    /// Drop the live prefix so the next `sync` forwards the whole prompt, and
    /// stop whatever the engine is still generating for it.
    fn invalidate(&mut self) {}
}

/// The end-of-sequence ids the correctness free-run reference stops on:
/// `ENGINE_EOS_IDS` when set, else the model's own EOS plus Nemotron's two
/// stop ids (`</s>` = 2 and `<|im_end|>` = 11, generation_config.json).
fn eos_ids(engine_eos: Option<i64>) -> Vec<i64> {
    let from_env = std::env::var("ENGINE_EOS_IDS")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|p| p.trim().parse::<i64>().ok())
                .collect::<Vec<_>>()
        })
        .filter(|v| !v.is_empty());
    let mut ids = from_env.unwrap_or_else(|| vec![2, 11]);
    if let Some(eos) = engine_eos {
        if !ids.contains(&eos) {
            ids.push(eos);
        }
    }
    ids
}

/// Session-backed engine for ONE phase. Cheap to mint: the session it borrows was
/// opened once at process start.
pub struct SessionEngine {
    session: Arc<Mutex<dyn EngineSession>>,
    /// The open free-run phase's resolved route + depth and the token the next
    /// cycle feeds. `None` until `free_decode_begin`.
    free_run: Option<FreeRun>,
}

struct FreeRun {
    route: Route,
    pending: i64,
}

impl SessionEngine {
    pub fn new(session: Arc<Mutex<dyn EngineSession>>) -> Self {
        Self {
            session,
            free_run: None,
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, dyn EngineSession + 'static> {
        self.session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn fault<T>(result: Result<T, String>) -> Result<T, EngineError> {
        result.map_err(EngineError::Fault)
    }

    /// Resolve a requested route + depth against the envelope AND against what
    /// the vendored engine implements. `Serial` is always runnable. `Mtp` needs
    /// the drafter armed at open (the leg booted with an mtp spec), and an explicit
    /// depth above [`ENGINE_IMPLEMENTED_DEPTH`] is refused by name.
    pub(crate) fn resolve(
        mtp_armed: bool,
        route: Route,
        requested_depth: Option<i64>,
    ) -> Result<i64, EngineError> {
        match route {
            Route::Serial => Ok(0),
            Route::Mtp => {
                if !mtp_armed {
                    return Err(EngineError::UnsupportedMode {
                        requested: "mtp".to_string(),
                        runnable: "serial (the resident was booted without the MTP drafter; \
                                   boot it with --spec mtp)"
                            .to_string(),
                    });
                }
                let depth = requested_depth.unwrap_or(ENGINE_IMPLEMENTED_DEPTH);
                if !(MTP_MIN_DEPTH..=MTP_MAX_DEPTH).contains(&depth) {
                    return Err(EngineError::DepthOutOfEnvelope {
                        requested: depth,
                        permitted: format!("{MTP_MIN_DEPTH}...{MTP_MAX_DEPTH}"),
                    });
                }
                if depth > ENGINE_IMPLEMENTED_DEPTH {
                    return Err(EngineError::DepthOutOfEnvelope {
                        requested: depth,
                        permitted: format!(
                            "{MTP_MIN_DEPTH}...{ENGINE_IMPLEMENTED_DEPTH} on this engine (the pinned \
                             engine drafts up to {ENGINE_IMPLEMENTED_DEPTH} tokens per cycle)"
                        ),
                    });
                }
                Ok(depth)
            }
        }
    }

    fn step_at_frontier(session: &mut dyn EngineSession) -> Result<Step, EngineError> {
        let token = session.argmax();
        let top_logits = session.top_logits(TOP_LOGITS_K);
        if top_logits.len() != TOP_LOGITS_K {
            return Err(EngineError::Fault(format!(
                "engine returned {} top logits, the gate needs exactly {TOP_LOGITS_K}",
                top_logits.len()
            )));
        }
        Ok(Step { token, top_logits })
    }
}

/// THE PHASE CLOSE. The adapter drops a phase's engine at the phase's
/// `phase_diagnostics` barrier (and on any fail-closed discard). benchd keeps
/// ONE worker, so one resident connection, for a whole leg, so the resident
/// sees no accept between phases; without this the phase's last request ran
/// on to the context limit with nobody reading it, and the GPU stayed busy
/// through benchd's quiescence gate before the next phase. Invalidating here
/// stops it the moment the phase is complete.
impl Drop for SessionEngine {
    fn drop(&mut self) {
        self.lock().invalidate();
    }
}

impl Engine for SessionEngine {
    fn drain_to_zero(&mut self) -> Result<u64, EngineError> {
        // One process per phase, so there is no allocator to drain. What a
        // phase start must guarantee is that its first sync does real work:
        // the engine's sync is free when the live prefix already equals the
        // prompt, so drop that prefix here.
        self.lock().invalidate();
        Ok(0)
    }

    fn prefill(&mut self, prompt: &[i64]) -> Result<i64, EngineError> {
        // A prefill phase wants the frontier token only. Bounding the sync
        // there means the engine is idle again when the reply comes back:
        // the next timed pass does not start behind a decode step, and the
        // GPU is quiet when benchd gates the next phase.
        let mut s = self.lock();
        Self::fault(s.sync_to_frontier(prompt))?;
        Ok(s.argmax())
    }

    fn decode_begin(&mut self, seed: &[i64]) -> Result<i64, EngineError> {
        let mut s = self.lock();
        Self::fault(s.sync(seed))?;
        Ok(s.argmax())
    }

    fn step(&mut self, input: i64) -> Result<Step, EngineError> {
        // TEACHER FORCING: the caller's token is evaluated; the model's own
        // choice is returned but never fed. Shared by decode_step AND
        // correctness_step; do NOT split.
        let mut s = self.lock();
        Self::fault(s.eval(input))?;
        Self::step_at_frontier(&mut *s)
    }

    fn correctness_begin(&mut self, prompt: &[i64]) -> Result<Step, EngineError> {
        let mut s = self.lock();
        Self::fault(s.sync(prompt))?;
        Self::step_at_frontier(&mut *s)
    }

    fn correctness_freerun(&mut self, prompt: &[i64], steps: i64) -> Result<Vec<i64>, EngineError> {
        // Free-run greedy REFERENCE generation: plain serial evals, stopping
        // where the golden stopped (an EOS id), never forcing an exact length.
        let mut s = self.lock();
        let eos = eos_ids(s.eos_token());
        Self::fault(s.sync_argmax_only(prompt))?;
        let mut out = Vec::with_capacity(steps.max(0) as usize);
        let mut next = s.argmax();
        for _ in 0..steps {
            out.push(next);
            if eos.contains(&next) {
                break;
            }
            Self::fault(s.eval(next))?;
            next = s.argmax();
        }
        Ok(out)
    }

    fn free_decode_begin(
        &mut self,
        seed: &[i64],
        route: Route,
        requested_depth: Option<i64>,
    ) -> Result<(i64, EffectiveSpec), EngineError> {
        let mut s = self.lock();
        let depth = Self::resolve(s.mtp_armed(), route, requested_depth)?;
        // The free run reads argmaxes only (the seed token, then committed
        // tokens and frontiers), never top-k logits.
        Self::fault(s.sync_argmax_only(seed))?;
        let seed_token = s.argmax();
        drop(s);
        self.free_run = Some(FreeRun {
            route,
            pending: seed_token,
        });
        let effective = EffectiveSpec {
            mode: route.as_str().to_string(),
            mtp: match route {
                Route::Serial => None,
                Route::Mtp => Some(EffectiveSpecMTP { depth }),
            },
        };
        Ok((seed_token, effective))
    }

    fn free_decode_run(&mut self, count: i64) -> Result<FreeRunResult, EngineError> {
        let (route, mut pending) = match self.free_run.as_ref() {
            Some(f) => (f.route, f.pending),
            None => {
                return Err(EngineError::Fault(
                    "free_decode_run with no open free-run phase".into(),
                ))
            }
        };
        if count <= 0 {
            return Err(EngineError::Fault(format!(
                "free_decode_run count {count} is not positive"
            )));
        }
        let mut tokens: Vec<i64> = Vec::with_capacity(count as usize);
        let mut acceptance_lengths: Vec<i64> = Vec::new();
        let (drafted_total, accepted_total, verify_replay_disagreements) = {
            let mut s = self.lock();
            match route {
                Route::Serial => {
                    for _ in 0..count {
                        Self::fault(s.eval(pending))?;
                        pending = s.argmax();
                        tokens.push(pending);
                        acceptance_lengths.push(1);
                    }
                    // The serial route reads no engine counters at all, so it
                    // reports no disagreement count rather than reporting one
                    // it never measured.
                    (0, 0, None)
                }
                Route::Mtp => {
                    let before = s.spec_counters();
                    // ONE REQUEST FOR THE WHOLE RUN when the session offers it
                    // (the resident does): the engine loops the cycles itself
                    // and hands back each cycle's committed tokens with the
                    // frontier argmax it left. The checks and the assembly
                    // below are the per-cycle loop's own, applied to those
                    // rounds, so both drives commit the same tokens and
                    // acceptance lengths. What goes away is the socket round
                    // trip after every cycle, which the GPU waits through idle.
                    let mut batched = match s.spec_run(pending, count) {
                        Some(rounds) => Some(Self::fault(rounds)?.into_iter()),
                        None => None,
                    };
                    while (tokens.len() as i64) < count {
                        let budget = count - tokens.len() as i64;
                        let (committed, frontier) = match batched.as_mut() {
                            Some(rounds) => {
                                let round = rounds.next().ok_or_else(|| {
                                    EngineError::Fault(format!(
                                        "speculative run ended after {} of {count} tokens",
                                        tokens.len()
                                    ))
                                })?;
                                (round.committed, Some(round.frontier))
                            }
                            None => (Self::fault(s.eval_speculative(pending, budget))?, None),
                        };
                        if committed.first() != Some(&pending) {
                            return Err(EngineError::Fault(format!(
                                "speculative cycle committed {committed:?}, expected it to \
                                 start with the fed token {pending}"
                            )));
                        }
                        // The cycle's outputs are every committed token after
                        // the fed one, plus the frontier argmax it left behind.
                        // The cycle contributes every committed token after
                        // the fed token, followed by the frontier argmax. That
                        // is exactly `committed.len()` outputs. Extend the
                        // destination directly instead of allocating and
                        // copying a second temporary Vec on every MTP round.
                        let produced_len = committed.len();
                        if (tokens.len() + produced_len) as i64 > count {
                            return Err(EngineError::Fault(format!(
                                "speculative cycle produced {} tokens with only {budget} wanted",
                                produced_len
                            )));
                        }
                        tokens.extend_from_slice(&committed[1..]);
                        pending = match frontier {
                            Some(frontier) => frontier,
                            None => s.argmax(),
                        };
                        tokens.push(pending);
                        acceptance_lengths.push(produced_len as i64);
                    }
                    if batched.is_some_and(|mut rounds| rounds.next().is_some()) {
                        return Err(EngineError::Fault(format!(
                            "speculative run ran more cycles than the {count} tokens wanted"
                        )));
                    }
                    let after = s.spec_counters();
                    let drafted = after.drafts.saturating_sub(before.drafts) as i64;
                    let accepted = after.hits.saturating_sub(before.hits) as i64;
                    let disagreements = after
                        .verify_replay_disagreements
                        .saturating_sub(before.verify_replay_disagreements)
                        as i64;
                    // The leg must run its declared configuration end to end.
                    // A quench, or a drafter that proposed nothing over a leg
                    // long enough to draft (the first cycle is the engine's
                    // plain baseline measurement), is a fault, not a result.
                    // The pinned resident enters no adaptive disable, so its
                    // counter never moves; the tripwire stays armed for an
                    // engine that does report one.
                    if after.quenches > before.quenches {
                        return Err(EngineError::Fault(
                            "the engine quenched speculation during the mtp leg".into(),
                        ));
                    }
                    if drafted == 0 && count >= 3 {
                        return Err(EngineError::Fault(format!(
                            "the mtp drafter proposed nothing over {count} tokens; the engine \
                             disabled its MTP block (see stderr) or the drafter is not armed"
                        )));
                    }
                    (drafted, accepted, Some(disagreements))
                }
            }
        };
        if let Some(f) = self.free_run.as_mut() {
            f.pending = pending;
        }
        Ok(FreeRunResult {
            committed_total: tokens.len() as i64,
            tokens,
            acceptance_lengths,
            drafted_total,
            accepted_total,
            verify_replay_disagreements,
        })
    }

    fn peak_ram_gb(&self) -> f64 {
        peak_rss_gb().unwrap_or(0.0)
    }
}

/// Peak resident set size of this process in GB, from `/proc/self/status`
/// (`VmHWM`). Audit only; benchd never scores it.
fn peak_rss_gb() -> Option<f64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
    let kib: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kib * 1024.0 / 1e9)
}

// =========================================================================== //
// The factory
// =========================================================================== //

/// Mints one [`SessionEngine`] per phase over the session opened at startup.
pub struct SessionFactory {
    session: Arc<Mutex<dyn EngineSession>>,
}

impl SessionFactory {
    pub fn with_session(session: Arc<Mutex<dyn EngineSession>>) -> Self {
        Self { session }
    }

    /// Attach to the window's resident engine over its Unix socket. NO LOAD
    /// HAPPENS HERE: the weights were loaded once when `tools/serve-up.sh`
    /// booted the resident, and this connects to that process. Returns the
    /// factory and the resident's own hello, whose `ident` is the engine
    /// identity benchd records.
    #[cfg(unix)]
    pub fn from_resident(
        socket_path: &str,
    ) -> Result<(Self, crate::resident::ResidentHello, std::time::Duration), String> {
        let session = crate::resident::ResidentSession::connect(socket_path)?;
        let hello = session.hello().clone();
        let connect = session.connect;
        Ok((
            Self::with_session(Arc::new(Mutex::new(session))),
            hello,
            connect,
        ))
    }
}

impl EngineFactory for SessionFactory {
    fn create(&self) -> Box<dyn Engine> {
        Box::new(SessionEngine::new(Arc::clone(&self.session)))
    }
}
