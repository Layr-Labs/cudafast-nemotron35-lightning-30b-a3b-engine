#!/usr/bin/env python3
"""stitch-prompts.py -- author one generation of stitched hidden prompts.

ORGANIZER TOOL (docs/pool-refresh-and-final-rescore.md, "Prompt generation").
Each prompt is paragraphs drawn at random from at least 4 different
public-domain works, in random order, trimmed to the fixture's seed_tokens with
the PINNED tokenizer (tools/author-prompts.py's own path). A leaked fragment
then identifies one paragraph, not the prompt.

    tools/stitch-prompts.py --corpus .mlxfast-private/corpus --snapshot <SNAPSHOT_DIR> \\
        --label g1 --count 8 --out .mlxfast-private/refresh/g1/prompts

writes, into --out (private: outside the repo, or under .mlxfast-private/):
  <label>-<n>.tokens.json   seed_tokens ids (what record-goldens.sh reads)
  <label>-<n>.txt           the text those ids encode
  provenance.json           per prompt: work id + paragraph index of every piece
and appends every paragraph it used to the ledger (default
.mlxfast-private/refresh/ledger.json). A paragraph in the ledger -- used by any
earlier generation or by `final` -- is never drawn again, and a label already in
the ledger is refused. Randomness is /dev/urandom (secrets.SystemRandom).

Corpus files are Project Gutenberg plain text (<work>.txt); only the text between
the "*** START OF" and "*** END OF" lines is used.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import secrets
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABEL_RE = re.compile(r"^(g[0-9]+|final)$")
MIN_WORDS, MAX_WORDS = 40, 400  # a paragraph, not a heading and not a chapter


def paragraphs(text: str) -> list[str]:
    """The Gutenberg body split into whitespace-normalized paragraphs (all of
    them, so an index is stable; filtering happens at draw time)."""
    lines = text.replace("\r\n", "\n").split("\n")
    start = next((i for i, ln in enumerate(lines) if ln.startswith("*** START OF")), None)
    end = next((i for i, ln in enumerate(lines) if ln.startswith("*** END OF")), None)
    if start is None or end is None or end <= start:
        raise ValueError("no *** START OF / *** END OF markers")
    body = "\n".join(lines[start + 1:end])
    return [" ".join(p.split()) for p in re.split(r"\n\s*\n", body) if p.strip()]


def para_key(p: str) -> str:
    return hashlib.sha256(p.encode()).hexdigest()


def load_corpus(corpus: Path) -> dict[str, list[str]]:
    works = {}
    for f in sorted(corpus.glob("*.txt")):
        try:
            works[f.stem] = paragraphs(f.read_text(encoding="utf-8", errors="replace"))
        except ValueError as e:
            print(f"stitch-prompts: skipping {f.name}: {e}", file=sys.stderr)
    return works


def stitch(works, label, count, used, rng, trim, min_works=4, attempts=200):
    """Build `count` prompts. works: {work: [paragraph]}; used: set of paragraph
    keys already spent (grown in place); rng: random.Random-like; trim(text) ->
    (ids, end_char) or ValueError when text is too short. Returns
    [(name, ids, text, pieces)] with pieces [{work, paragraph, sha256, truncated}]."""
    out = []
    for n in range(1, count + 1):
        for _ in range(attempts):
            avail = {w: [i for i, p in enumerate(ps)
                         if MIN_WORDS <= len(p.split()) <= MAX_WORDS and para_key(p) not in used]
                     for w, ps in works.items()}
            avail = {w: ix for w, ix in avail.items() if ix}
            if len(avail) < min_works:
                raise ValueError(f"only {len(avail)} works still have unused paragraphs; need {min_works}")
            chosen = rng.sample(sorted(avail), rng.randint(min_works, min(min_works + 2, len(avail))))
            picks, taken = [], set()
            result = None
            while result is None:
                grew = False
                for w in chosen:  # one more paragraph from each chosen work
                    left = [i for i in avail[w] if (w, i) not in taken]
                    if left:
                        i = rng.choice(left)
                        taken.add((w, i))
                        picks.append((w, i))
                        grew = True
                if not grew:
                    break
                order = picks[:]
                rng.shuffle(order)
                texts = [works[w][i] for w, i in order]
                joined = "\n\n".join(texts)
                try:
                    ids, end = trim(joined)
                except ValueError:
                    continue  # still short of seed_tokens: another round
                pieces, pos = [], 0
                for (w, i), t in zip(order, texts):
                    if pos >= end:
                        break
                    pieces.append({"work": w, "paragraph": i, "sha256": para_key(t),
                                   "truncated": pos + len(t) > end})
                    pos += len(t) + 2
                result = (ids, joined[:end], pieces)
            if result and len({p["work"] for p in result[2]}) >= min_works:
                break
        else:
            raise ValueError(f"could not stitch {label}-{n} from >= {min_works} works in {attempts} attempts")
        ids, text, pieces = result
        used.update(p["sha256"] for p in pieces)
        out.append((f"{label}-{n}", ids, text, pieces))
    return out


def private_path(p: Path) -> Path:
    """Refuse a path inside the repo tree unless it is under .mlxfast-private/."""
    p = p.resolve()
    if p.is_relative_to(ROOT) and not p.is_relative_to(ROOT / ".mlxfast-private"):
        raise ValueError(f"{p} is inside the repository; private material goes outside it or under .mlxfast-private/")
    return p


def load_ledger(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {"labels": [], "paragraphs": {}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--snapshot", required=True, help="the verified target snapshot (holds tokenizer.json)")
    ap.add_argument("--label", required=True, help="g<N> or final")
    ap.add_argument("--count", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ledger", default=str(ROOT / ".mlxfast-private/refresh/ledger.json"))
    ap.add_argument("--fixture", default=str(ROOT / "fixtures/nemotron35_lightning_30b_a3b_track.json"))
    a = ap.parse_args()

    def die(msg):
        print(f"stitch-prompts: {msg}", file=sys.stderr)
        return 1

    if not LABEL_RE.match(a.label):
        return die(f"label '{a.label}' is not g<N> or final")
    try:
        out, ledger_path = private_path(Path(a.out)), private_path(Path(a.ledger))
    except ValueError as e:
        return die(str(e))
    ledger = load_ledger(ledger_path)
    if a.label in ledger["labels"]:
        return die(f"generation '{a.label}' is already in the ledger {ledger_path}; a generation is authored once")
    if list(out.glob(f"{a.label}-*.tokens.json")):
        return die(f"{out} already holds {a.label} prompts")

    spec = importlib.util.spec_from_file_location("author_prompts", ROOT / "tools/author-prompts.py")
    ap_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ap_mod)
    try:
        tok, seed, vocab = ap_mod.load_pinned_tokenizer(a.snapshot, a.fixture)
    except ValueError as e:
        return die(str(e))

    works = load_corpus(Path(a.corpus))
    used = set(ledger["paragraphs"])
    try:
        prompts = stitch(works, a.label, a.count, used, secrets.SystemRandom(),
                         lambda text: ap_mod.seed_ids(tok, text, seed, vocab))
    except ValueError as e:
        return die(str(e))

    out.mkdir(parents=True, exist_ok=True)
    prov = {"label": a.label, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "seed_tokens": seed, "prompts": {}}
    for name, ids, text, pieces in prompts:
        (out / f"{name}.tokens.json").write_text(json.dumps(ids))
        (out / f"{name}.txt").write_text(text, encoding="utf-8")
        prov["prompts"][name] = pieces
        for p in pieces:
            ledger["paragraphs"][p["sha256"]] = {"label": a.label, "work": p["work"], "paragraph": p["paragraph"]}
        print(f"stitch-prompts: wrote {name} ({len(ids)} tokens, {len({p['work'] for p in pieces})} works)")
    (out / "provenance.json").write_text(json.dumps(prov, indent=1))
    ledger["labels"].append(a.label)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = ledger_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(ledger, indent=1))
    os.replace(tmp, ledger_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
