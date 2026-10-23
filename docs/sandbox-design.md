# Sandboxing participant code on the ranked box

Status: implemented. The ranked workflow (`.github/workflows/benchmark.yml`),
`tools/engine/build.sh`, `tools/serve-up.sh`, `tools/stage-cuda-engine.sh`,
`tools/run-engine-sandboxed.sh`, `tools/sandbox-wipe.sh` and
`tools/box-runner/setup-sandbox-user.sh` carry it out; the ranked preflight
refuses a box where it is not in place.

## 1. Threat model

A ranked job runs as the box user `runner`. `runner` is in the `docker` group
(root-equivalent: any `docker run -v /:/host` is root on the box) and in
`bench` (the GPU lock and the private goldens), and it holds the runner's
credentials. A participant controls the code in the editable paths and may be
malicious. Without a sandbox, that code would run with everything `runner` can
do: read the private goldens and the hidden prompts' answers, rewrite the
trusted tools in the job checkout and forge a score, persist on the box, or
reach the runner's credentials.

## 2. Where participant code runs

| # | What | Where it runs | As |
|---|---|---|---|
| E1 | `cargo build` of `harness/protocol-adapter` (build.rs, proc macros), driven by `tools/engine/build.sh` | host, offline, under `prlimit` | `nemosub` |
| E2 | `harness/kernels/build.sh` | pinned container, `--network none`, checkout read-only, resource limits, `--cap-drop ALL` | `nemosub` uid |
| E3 | `harness/vllm_resident/resident.py` + the `vllm/` overlay (candidate leg) | pinned container, checkout read-only, `--network none`, GPU | `nemosub` uid |
| E4 | the built adapter (`cuda-engine`), spawned by benchd for the candidate leg | host, via `tools/run-engine-sandboxed.sh`, under `prlimit --fsize` | `nemosub` |

Everything else on the box (preflight, setup, serve-up, measure-and-score,
benchd, the sealed score) is trusted code from the base commit and runs as
`runner`. The control leg runs the pinned reference unchanged, as `runner`.

## 3. What the sandbox user cannot reach

`nemosub` is not in `docker`, `bench` or `sudo`, and cannot write the checkout.
As a result, participant code cannot:

- read the private goldens (`root:bench`) or the runner's credentials;
- write the trusted tools in the job checkout (read-only ACL) or anything the
  measuring side later executes;
- reach the network at all, loopback and the host resolver included;
- schedule anything that outlives the job (`cron` and `at` are denied, and
  every `nemosub` process is killed and every `nemosub` file wiped on the
  schedule in section 5 item 6);
- write a large file or carry output into the job log (its stderr goes to a
  `runner`-private file under `prlimit --fsize`);
- forge the score: benchd, another uid, seals `score.json`, and the router
  accepts only exactly one benchd marker from a green measurement step.

The residual risk is a container or kernel escape, the same as on any shared
GPU host.

## 4. Options considered

**A. bubblewrap around E1 and E4.** Rejected: DGX OS sets
`kernel.apparmor_restrict_unprivileged_userns=1`, and `bwrap` fails with
"setting up uid map: Permission denied" (tested on a Spark box). Making it work
needs an AppArmor profile on every box and still leaves E2.

**B. Everything participant-built runs in containers.** Build the adapter in a
container, and run the adapter in a container too. Rejected for E4: benchd
talks to the adapter over stdio, and `docker run -i` proxies stdio through the
container shim on every message, which adds latency inside the timed decode
window and differs from the control leg.

**C. A second, unprivileged user (chosen).** Create `nemosub`: no
`docker`, no `bench`, no sudo, no write access to the checkout. Every
untrusted execution point runs as `nemosub`; everything trusted stays
`runner`. It needs no namespaces, so AppArmor is not in the way, and it adds
nothing to the timed path (a native process with plain pipes).

## 5. The design (C)

1. **Build (E1).** `runner` runs `cargo vendor --locked` from the repo root
   into a per-job vendor directory (vendoring runs no build scripts), owned by
   `runner` and read-only to `nemosub`. Then
   `sudo -n -u nemosub cargo build --offline --locked` with a `nemosub`-owned
   `CARGO_HOME` and target directory, using the vendored sources. `runner`'s
   own Cargo state is never shared with `nemosub`, so nothing `nemosub` writes
   can be picked up by a later `runner` Cargo run. Because `runner` vendors
   from participant manifests, vendoring is refused unless every `Cargo.lock`
   source is crates.io, every `path` in a `harness/**/Cargo.toml` resolves
   inside `harness/`, and no `[patch]`, `[replace]` or other registry is
   named; a vendored tree holding a symlink or special file is refused too.
   The `nemosub` build runs under `prlimit` (processes, address space). Cargo is always invoked from
   the repo root (as `tools/engine/build.sh` does), so a
   `harness/**/.cargo/config.toml` is never read. `runner` copies the binary
   into the trusted staging path, which `nemosub` cannot write, without
   following links, and re-checks the copy (a regular file; for the kernels
   output, no symlink or special file anywhere), so a swap after the check is
   refused. After the build, `runner` removes all files from `cargo-home`,
   `target` and `kernels-out`. These directories then have mode `2750`.
   The staged adapter, vLLM overlay and kernels stay under `.build`.
   `runner` owns these staged files. The files have no group or other write
   permission. The build needs a Rust toolchain that `nemosub` can use.
   The shared read-only `/opt/rust` has the same version as `runner`'s toolchain.
2. **Kernels (E2).** The container runs `--user nemosub`, mounts the checkout
   **read-only**, and mounts only a `nemosub`-owned output directory
   read-write, outside the checkout. It gets CPU, memory and process limits
   (`--cpus`, `--memory`, `--pids-limit`), `--cap-drop ALL` and
   `--security-opt no-new-privileges` (always, no off switch), and the box must be quiet again
   before setup continues. This closes the "rewrite the trusted tools" hole on
   its own.
3. **Resident (E3), candidate leg only.** `serve-up.sh` runs the candidate's
   container as `nemosub`. The workflow names the candidate checkout in
   `SERVE_UP_SANDBOX_TREE`. benchd boots both legs with one environment, so
   only the tree at that path takes `SANDBOX_USER` as `SERVE_UP_CONTAINER_USER`.
   The reference tree runs its resident as `runner`, because benchd's
   control-leg worker runs as `runner` and must open its socket. Sockets are split per leg: the base socket directory stays
   `runner`-owned and not writable by `nemosub` (mode `0711`, so `nemosub` can
   only reach a path it is given); each candidate leg gets its own subdirectory
   that `nemosub` can write. The control leg's socket, and every `.pid`
   sidecar, stay where `nemosub` cannot write, so a leftover candidate process
   cannot swap the control socket or steer `--stop` into signalling a `runner`
   process. The resident log dir (`SERVE_UP_LOG_DIR`, `/var/lib/nemotron35/serve-log` on
   a box) is `runner:runner` 0750 (setup strips other-access from what is
   already there) and `serve-up.sh` creates each resident log 0600 and the
   identity file 0640, so a later job's `nemosub` cannot read what an earlier
   candidate resident printed. The resident chmods its socket `0600`, so its health probe
   and the adapter must be `nemosub` too: the probe runs
   via `sudo -n -u nemosub`. The control leg runs the pinned reference's own
   `serve-up.sh`, which ignores the new variable, so it keeps running as
   `runner` — trusted code, unchanged, so its calibration stays valid.
4. **Adapter (E4), candidate leg only.** benchd keeps getting the normal
   staged path `.build/release/mlxfast-runtime-worker` as `--engine`, because
   it resolves the control leg's engine at that same root-relative path inside
   the reference workspace. With `SANDBOX_USER` set, `stage-cuda-engine.sh`
   installs the adapter beside it as `mlxfast-runtime-worker.sandboxed` and
   writes a trusted shim at the staged path. benchd spawns the engine with a
   cleared environment, so the shim bakes in, as literal exports, a fixed
   `PATH`, `SANDBOX_USER`, `SANDBOX_ROOT`, `SANDBOX_JOB` (charset-checked),
   `SANDBOX_ENGINE_LOG_DIR` (the job's `RUNNER_TEMP`, which must be a
   runner-owned directory) and `SANDBOX_ENGINE_BIN`, then execs the trusted
   wrapper `tools/run-engine-sandboxed.sh`; `measure-and-score.sh` refuses to run if
   the staged engine is not exactly that shim. The wrapper execs the adapter with
   `sudo -n -u nemosub --preserve-env=DS4_RESIDENT_SOCKET,BENCH_WORKER_RESIDENT_SOCKET`.
   stdio to benchd is inherited, so the protocol path is identical to an
   unsandboxed engine once the process is up. sudo adds a few milliseconds at spawn, outside every
   timed window. The adapter's **stderr goes to a `runner`-private file**, not
   the job log: the engine necessarily sees the hidden prompts, and the job log
   must not become a way to carry them out. The adapter runs under
   `prlimit --fsize` (64 MiB), `--nproc` (4096 by default) and `--as` (16 GiB
   by default), so neither that file nor any file it writes can grow past the
   fsize cap, and fork or address-space bombs are bounded; stdio pipes are not
   files and are unaffected. Override the process and address-space caps with
   `ADAPTER_NPROC_LIMIT` and `ADAPTER_AS_LIMIT` (bytes for `--as`) when
   testing locally; the ranked workflow does not set them. When the adapter exits, the
   wrapper kills **every** `nemosub` process (`sudo -n -u nemosub kill -KILL -1`)
   and checks none remain, so nothing the candidate started can run during the
   next control leg. The quiet-box check (load, GPU utilisation, memory) runs
   again before each control leg, not only before the first.
5. **Network.** `nemosub` on the host (E1 build, E4 adapter) could reach the
   network. Block it: `nemosub` sends no TCP or UDP at all, **loopback
   included** (it only needs the resident's Unix socket), for both IPv4 and
   IPv6 (`iptables` and `ip6tables -m owner --uid-owner nemosub`). Loopback is
   not exempt because the host resolver (`127.0.0.53`) forwards names, which
   is a way out. Kernel netfilter rules are not persistent across reboot unless
   `iptables-persistent` is installed, so a boot-time oneshot unit
   (`nemotron35-sandbox-egress.service`) re-applies the block; the box-runner
   unit refuses to start without it. The resolver's Unix sockets (`systemd-resolved`'s varlink
   socket, `nscd`'s) are denied to `nemosub` by ACL; `/run` is a tmpfs, so a
   boot-time oneshot unit (`nemotron35-sandbox-acl.service`) re-applies it,
   and `--check` proves `getent hosts` fails as `nemosub`. The containers run
   `--network none` and are unaffected.
6. **What `nemosub` can still do.** Read world-readable files (public repo,
   public weights), read the job checkout (a per-job ACL, `rX` on the checkout
   and `--x` on the directories above it up to `runner`'s home, which is not
   world-readable; never `RUNNER_TEMP`, which gets an explicit deny). `--x` on
   a directory still opens every world-readable entry in it whose name is
   guessed, so every entry directly under `runner`'s home, `actions-runner`,
   `_work` and the job's workspace dir, other than the one component leading
   to the checkout, gets `u:nemosub:---` (setup once as root, the workflow
   again per job): `actions-runner/.env`, `_diag`, `.credentials*`, `~/.cache`
   stay out of reach. **Every file `runner` writes under `GITHUB_WORKSPACE`
   inherits the checkout's default ACL and is `nemosub`-readable: never write a
   secret, a golden or a hidden prompt there; runner-private files go in
   `RUNNER_TEMP`.** It can also fill
   `/tmp` (a nuisance, not a leak). Nothing it leaves may reach the next leg or
   job. The passwd home is root-owned and mode `0755`. It has no ACL or xattr.
   The sandbox user cannot write it. The resident uses its per-leg run home.
   The adapter uses the per-job `adapter-home`.
   `tools/sandbox-wipe.sh` deletes every `nemosub`-owned file in every
   directory it can write (the box's wipe list: every world-writable directory
   on a local filesystem, so `/tmp`, `/var/tmp`, `/run/lock`, and `/var/crash`).
   It also deletes owned files below the passwd home. The script always adds
   `/dev/shm` and `/dev/mqueue` to the list. It reads each SysV IPC table in
   `/proc/sysvipc`. It selects an object when its numeric `uid` or `cuid`
   equals the sandbox uid. If `/proc/sysvipc` is absent, it uses `ipcs -c`.
   An unreadable table or a failed command makes the wipe fail. The script
   removes each object as `nemosub` first. Existing root sudo access is the
   fallback. The script does not add sudo rights. It verifies that no selected
   file or IPC object remains. It also checks foreign-owned writable paths.
   It checks each wipe-list directory and the per-job directory.
   A hit fails the wipe.
   The error gives only a count. It never gives a path name. It runs:
   - at **job start** and **job end** (the ranked workflow);
   - **before the timed measurement**, when `SANDBOX_USER` is set:
     `tools/nemotron35-measure-and-score.sh` kills every `nemosub` process,
     runs the wipe, then `tools/wait-for-memory.sh`. It also verifies an empty
     `adapter-home` and `sock`. It refuses staged build output that is not
     runner-owned or has a group or other write bit;
   - **after every candidate leg**, when `serve-up.sh --stop` tears down a
     sandboxed resident (`SERVE_UP_CONTAINER_USER` is set or derived from
     `SERVE_UP_SANDBOX_TREE`, or the socket path
     lies under `$SANDBOX_ROOT/$SANDBOX_JOB/sock/`): after the container is
     gone, the same kill-and-verify step runs. The wipe runs next. The stop
     recreates and verifies `adapter-home`, and empties all directories under
     `sock`. The wipe also verifies the passwd home metadata.
     Then `tools/wait-for-memory.sh` runs. The stop fails if any step fails;
   - **between the two control-leg retry attempts**, when
     `measure-and-score.sh` discards attempt 1: kill, wipe, wait for memory,
     then empty the five per-job directories the sandbox user writes.
   The adapter's `HOME` and `TMPDIR` are the per-job `adapter-home`. Nothing it
   schedules may outlive `kill -1`: `nemosub` is in `/etc/cron.deny` and
   `/etc/at.deny` and does not linger. Its output does
   not reach the job log (item 4). Job logs must still never be shown to
   participants: only the score is returned to them. A printed fake score line cannot pass: the router requires
   exactly one benchd marker and a green measurement step, and `nemosub`
   cannot signal or trace benchd (another uid).

The runner's credentials, the goldens, the trusted tools and the box itself
are therefore out of reach of participant code. The residual risk is a container or kernel
escape, the same as any shared GPU host.

## 6a. Shared contract (every task uses exactly these names)

The names below are the defaults. Section 6b gives the environment variable
that changes each name for a second runner on the same box.

| Thing | Value |
|---|---|
| Sandbox user | `nemosub`: system user, shell `/usr/sbin/nologin`, passwd home `/var/lib/nemosub` (`root:root` `0755`, no ACL, no xattr), NOT in `docker`, `bench` or `sudo` |
| Shared group | `nemoshare`: members `runner` and `nemosub` |
| Sandbox root | `/var/lib/nemotron35/sandbox`, owner `runner:nemoshare`, mode `2750` (nemosub can traverse and read, not write) |
| Per-job dirs (created by `runner` under the root, removed at job end) | `vendor` and `sock` start as `2750`. `cargo-home`, `target` and `kernels-out` start as `2770` for the build. The Rust build uses `cargo-home` for `HOME` and `CARGO_HOME`. Setup empties them and changes them to `2750`. `adapter-home` stays `2770`. The adapter uses it for `HOME` and `TMPDIR`. `serve-up.sh` creates each `sock/<tag>/` as `2770`. Each sandboxed stop recreates `adapter-home` with its owner, group, mode and default ACL. It empties `sock`. |
| Rust for nemosub | `/opt/rust/cargo` + `/opt/rust/rustup`, root-owned, world-readable, same toolchain as `runner`'s (`rustup show active-toolchain` as runner); callers set `PATH=/opt/rust/cargo/bin:...` AND `RUSTUP_HOME=/opt/rust/rustup` |
| sudoers | `/etc/sudoers.d/nemotron35-sandbox`: `runner ALL=(nemosub) NOPASSWD:SETENV: ALL` |
| Drop to sandbox | `sudo -n -u nemosub --preserve-env=VAR1,VAR2 -- cmd …` (never a shell string) |
| Kill every sandbox process | `sudo -n -u nemosub -- kill -KILL -1`, then verify `pgrep -u nemosub` is empty |
| Egress | `iptables` and `ip6tables`: `OUTPUT -m owner --uid-owner nemosub -j REJECT` (no loopback exemption; setup deletes the older `! -o lo` rule); re-applied at boot by `nemotron35-sandbox-egress.service` |
| Resolver | ACL `u:nemosub:---` on `/run/systemd/resolve/io.systemd.Resolve` and `/run/nscd/socket` when present, re-applied at boot by `nemotron35-sandbox-acl.service` |
| No persistence | `nemosub` in `/etc/cron.deny` and `/etc/at.deny`; no crontab, no at job, no linger |
| Wipe list | `/etc/nemotron35-sandbox/wipe-dirs`, root 0644, one directory per line: every world-writable directory on a local filesystem that `nemosub` can enter and write, plus `/dev/shm` and `/dev/mqueue`; written by `setup-sandbox-user.sh` (re-run it after adding a mount) |
| Wipe | `tools/sandbox-wipe.sh` removes owned files and SysV IPC with a matching `uid` or `cuid`. It clears the sandbox user's persistent and user keyrings. It verifies the passwd home. It rejects foreign-owned writable paths in wipe-list and per-job directories. It always includes `/dev/shm` and `/dev/mqueue`. Enumeration errors fail the wipe. The workflow runs it at job start and end. The measure script runs it before measurement and between retries. Candidate `serve-up.sh --stop` also recreates `adapter-home` and empties `sock`. |
| Adapter limits | `prlimit --fsize=67108864 --nproc` (default 4096, `ADAPTER_NPROC_LIMIT`) `--as` (default 16 GiB, `ADAPTER_AS_LIMIT` in bytes) on the adapter in `tools/run-engine-sandboxed.sh`; the build uses separate `CARGO_NPROC_LIMIT` / `CARGO_AS_LIMIT` |
| Resident logs | `SERVE_UP_LOG_DIR` (from `actions-runner/.env`, else `/var/lib/nemotron35/serve-log`) `runner:runner` 0750, nothing in it other-readable; `serve-up.sh` writes resident logs 0600, the identity file 0640 |
| Checkout access | per job, as `runner`: `setfacl -R -P -m u:nemosub:rX` (and `-d`) on `$GITHUB_WORKSPACE`, `--x` on each parent up to `runner`'s home, `---` on every other entry directly under each of those parents (all but the component leading to the checkout), on `RUNNER_TEMP` and on `_work/_actions` and `_work/_tool` when present; setup gives `--x` on `runner`'s home, `actions-runner`, `_work` and `---` on every entry under the first two but the next one down and on `_work/_*`. Files `runner` writes under the checkout are `nemosub`-readable (default ACL): never a secret there |
| Workflow env | `SANDBOX_USER` and `SANDBOX_ROOT` from the runner environment, else `nemosub` and `/var/lib/nemotron35/sandbox` (the first step of the ranked job resolves them into `GITHUB_ENV`), `SANDBOX_JOB=<run_id>-<run_attempt>`, `SERVE_UP_SANDBOX_TREE=$GITHUB_WORKSPACE` in the measurement step: only the candidate checkout's serve-up takes `SANDBOX_USER` as its container user, and the reference tree's serve-up runs as `runner` |
| Box check | `tools/box-runner/setup-sandbox-user.sh --check` exits 0 only if all of the above holds |

benchd starts `tools/serve-up.sh --stop` with its own environment. It does not
call `env_clear` in `crates/benchd/src/legserve.rs`. Thus, `SANDBOX_*` and
`SERVE_UP_SANDBOX_TREE` reach the stop path on candidate legs. If a stop fails
after a successful measurement, benchd seals `LEG-TEARDOWN-FAILED` and refuses
the run. The driver reports exit code 75 without a new measured attempt. The
router classifies this result as infrastructure. The job-end cleanup is the
final backstop. benchd also checks load and GPU use before each timed phase.
Its quiescing gate and the kill check in item 4 form the per-leg quiet check.

## 6. Operator steps

1. On each box, once, as root: `sudo tools/box-runner/setup-sandbox-user.sh`.
   It creates `nemosub`, the `/opt/rust` toolchain, the directories, the
   sudoers drop-in, the egress block unit, the resolver ACL unit, the cron/at deny
   entries and the wipe list, and is idempotent. Re-run it after adding a
   mount.
2. `tools/box-runner/setup-sandbox-user.sh --check` verifies all of section 6a
   without changing anything. The ranked preflight runs it and refuses the job
   if it fails.
3. The workflow sets the variables in the "Workflow env" row, applies the
   checkout ACLs and runs `tools/sandbox-wipe.sh` at job start and in cleanup.
   The trusted measure-and-score and candidate serve-up steps run the same wipe
   (with a preceding kill where needed) at the points listed in the Wipe row;
   no per-job operator action is needed.

## 6b. A second runner on one box

Two runners can serve jobs on one box. Each runner has its own runner account,
sandbox user, share group, units and files. The GPU lock
`/tmp/mtplx-gpu-exclusive.lock` is shared, so the two runners never measure at
the same time. The `cuda-setup-scripts` `gpu-lock` unit creates it.
The lock is `root:bench` and mode `0660`. A ranked job never creates it.
The sandbox user is not in `bench` and cannot write the lock.
A job of one runner never kills, wipes or reads the processes
and files of the other runner's sandbox user.

Each name in section 6a comes from one environment variable. If a variable is
not set, the tools use the section 6a value. A box with one runner sets none of
them.

| Variable | Default | Second runner (example) | Read by |
|---|---|---|---|
| `RUNNER_USER` | `runner` | `layrbench` | `setup-sandbox-user.sh` |
| `SANDBOX_USER` | `nemosub` | `layrsub` | setup, workflow, every sandbox tool |
| `SHARE_GROUP` | `nemoshare` | `layrshare` | setup, `tools/engine/build.sh`, `tools/serve-up.sh` |
| `SANDBOX_ROOT` | `/var/lib/nemotron35/sandbox` | `/var/lib/layrbench/sandbox` | setup, workflow |
| `SANDBOX_ETC` | `/etc/nemotron35-sandbox` | `/etc/layrbench-sandbox` | setup (`wipe-dirs`, `egress.sh`), `tools/sandbox-wipe.sh` |
| `SUDOERS_NAME` | `nemotron35-sandbox` | `layrbench-sandbox` | setup (a file in `/etc/sudoers.d`; no dot in the name) |
| `EGRESS_UNIT` | `nemotron35-sandbox-egress.service` | `layrbench-sandbox-egress.service` | setup |
| `ACL_UNIT` | `nemotron35-sandbox-acl.service` | `layrbench-sandbox-acl.service` | setup |
| `RUST_DIR` | `/opt/rust` | `/opt/layrbench-rust` | setup, `tools/engine/build.sh` |
| `SERVE_UP_LOG_DIR` | `/var/lib/nemotron35/serve-log` (setup) | `/var/lib/layrbench/serve-log` | setup (from the runner's `actions-runner/.env`), `tools/serve-up.sh` |
| `SERVE_UP_CONTAINER_PREFIX` | `serve-up-` | `layr-serve-up-` | `tools/serve-up.sh`, `tools/box-runner/supervisor.sh` |

The sandbox home is always `/var/lib/<SANDBOX_USER>`.

To add the second runner:

1. Write each value as one `NAME=value` line in the second runner's
   `actions-runner/.env`. Every job of that runner gets them. Restart the
   runner service after a change.
2. As root, run the setup with the same values:
   `sudo env RUNNER_USER=layrbench SANDBOX_USER=layrsub SHARE_GROUP=layrshare ... tools/box-runner/setup-sandbox-user.sh`.
3. As the second runner account, run the check with the same values. The
   ranked preflight runs it again for each job.

The setup of the second runner adds its own group, user, directories, sudoers
file, units, `iptables` rules, wipe list and toolchain. It removes or replaces
only rules for its own sandbox user. In the shared files `/etc/cron.deny`,
`/etc/at.deny` and the resolver socket ACLs, it adds an entry for its own
sandbox user and changes no other entry.

The workflow resolves `SANDBOX_USER` and `SANDBOX_ROOT` in its first step,
because a step `env:` block cannot read the runner process environment. The
tools read the other names from the runner environment. The workflow and
`tools/` are outside `editablePaths`, so a submission cannot change a name.

## 7. Adversarial review findings

The design was reviewed from the point of view of a malicious participant.
These attacks were found and are closed above: leftover candidate processes
or `/dev/shm` fill between legs (kill and wipe after each candidate
`serve-up --stop`, before the timed measurement, and between retries; re-check
quiet before each control leg); shared socket directory races and
`.pid` sidecar steering (per-leg directories, sidecars out of `nemosub`'s
reach); prompt leakage through the job log (candidate stderr to a private file;
job logs never shown to participants); shared Cargo state flowing back to
`runner` (per-job `cargo vendor`, private Cargo homes); unbounded kernel builds
and IPv6 egress (container limits; `ip6tables` too); a build that runs as
`runner` (the preflight refuses a box without the sandbox).

## 8. How we know it works

1. `setup-sandbox-user.sh --check` passed on all 8 boxes (2026-09 and 2026-10 checks; the contest fleet is now 5).
2. Hostile submissions (the router's load-generator variants) fail every
   attack, including a candidate that
   leaves a busy process behind and one that prints prompt tokens to stderr.
3. **Fairness is unchanged:** 20 null submissions across the fleet score within
   the load test's distribution (mean 0.9959, sd 0.0058). A shift would mean
   the sandbox touched timing.
4. The legitimate variants from the load test (MTP, CUDA graphs, garbage) end
   exactly as before.
