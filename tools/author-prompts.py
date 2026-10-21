#!/usr/bin/env python3
"""author-prompts.py -- turn organizer texts into seed-length prompt files.

ORGANIZER TOOL. The timed pool and the hidden correctness prompt are private
organizer material; their source texts never enter this repository. This turns
each text into exactly the fixture's `seed_tokens` token ids with the PINNED
tokenizer of the staged target snapshot. No BOS is added (the pinned
tokenizer_config.json sets add_bos_token false, and benchd feeds token ids
straight to the engine).

Two forms:

  * default: the text is raw continuation text, and the first `seed_tokens`
    ids are kept. A text too short to fill the seed is refused.
  * --chat: the text is ONE user message. It is rendered through the PINNED
    chat_template.jinja of the snapshot, with the generation prompt and
    thinking off, and the whole rendering is the prompt. A rendering that is
    not exactly `seed_tokens` ids is refused: a chat prompt is never trimmed.

    tools/author-prompts.py --snapshot <SNAPSHOT_DIR> --out <DIR> [--chat] NAME=TEXT_FILE ...

writes <DIR>/<NAME>.tokens.json (a JSON array of ids) for each pair. The
`tokenizers` and `jinja2` packages are in the pinned engine image; run this
there if the host lacks them:

    docker run --rm --network none -v "$PWD:$PWD" -w "$PWD" --entrypoint python3 \\
      <image@digest> tools/author-prompts.py --snapshot ... --out ... botany=...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

def load_pinned_tokenizer(snapshot: str, fixture_path: str):
    """(tokenizer, seed_tokens, vocab_size) -- the PINNED tokenizer.json of the
    snapshot, its digest checked against the fixture. ValueError on a mismatch."""
    from tokenizers import Tokenizer  # the pinned image carries it

    tok_path = os.path.join(snapshot, "tokenizer.json")
    fixture = json.load(open(fixture_path))
    pin = next((f for f in fixture["target"]["files"] if f["path"] == "tokenizer.json"), None)
    if pin is None:
        raise ValueError("the fixture pins no tokenizer.json")
    digest = hashlib.sha256(open(tok_path, "rb").read()).hexdigest()
    if digest != pin["sha256"]:
        raise ValueError(f"{tok_path} is sha256 {digest}, the fixture pins {pin['sha256']}")
    # The fixture is the one declaration of the seed length (benchd reads the same field).
    return Tokenizer.from_file(tok_path), int(fixture["seed_tokens"]), fixture["target"]["vocab_size"]


def pinned_file(snapshot: str, fixture_path: str, rel: str) -> str:
    """The text of a snapshot file, its digest checked against the fixture pin."""
    path = os.path.join(snapshot, rel)
    fixture = json.load(open(fixture_path))
    pin = next((f for f in fixture["target"]["files"] if f["path"] == rel), None)
    if pin is None:
        raise ValueError(f"the fixture pins no {rel}")
    data = open(path, "rb").read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != pin["sha256"]:
        raise ValueError(f"{path} is sha256 {digest}, the fixture pins {pin['sha256']}")
    return data.decode("utf-8")


def chat_ids(tok, template: str, user: str, seed_tokens: int, vocab: int):
    """The ids of ONE user message rendered through the chat template, with the
    generation prompt and thinking off. ValueError unless exactly seed_tokens."""
    from jinja2.sandbox import ImmutableSandboxedEnvironment  # the pinned image carries it

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = lambda value, **_: json.dumps(value)
    rendered = env.from_string(template).render(
        messages=[{"role": "user", "content": user}], add_generation_prompt=True, enable_thinking=False)
    ids = tok.encode(rendered, add_special_tokens=False).ids
    if len(ids) != seed_tokens:
        raise ValueError(f"the chat rendering is {len(ids)} tokens; the seed is exactly {seed_tokens}")
    assert all(0 <= t < vocab for t in ids)
    return ids


def seed_ids(tok, text: str, seed_tokens: int, vocab: int):
    """(the first seed_tokens ids of text, the char offset where they end).
    Raw continuation text: no special tokens. ValueError when text is short."""
    enc = tok.encode(text, add_special_tokens=False)
    if len(enc.ids) < seed_tokens:
        raise ValueError(f"{len(enc.ids)} tokens; the seed needs {seed_tokens}")
    ids = enc.ids[:seed_tokens]
    assert all(0 <= t < vocab for t in ids)
    return ids, enc.offsets[seed_tokens - 1][1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", required=True, help="the verified target snapshot (holds tokenizer.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fixture", default=os.path.join(os.path.dirname(__file__), "..", "fixtures",
                                                      "nemotron35_lightning_30b_a3b_track.json"))
    ap.add_argument("--chat", action="store_true",
                    help="each text is one user message, rendered through the pinned chat template")
    ap.add_argument("pairs", nargs="+", metavar="NAME=TEXT_FILE")
    args = ap.parse_args()

    try:
        tok, seed_tokens, vocab = load_pinned_tokenizer(args.snapshot, args.fixture)
        template = pinned_file(args.snapshot, args.fixture, "chat_template.jinja") if args.chat else None
    except ValueError as e:
        print(f"author-prompts: {e}", file=sys.stderr)
        return 1

    os.makedirs(args.out, exist_ok=True)
    for pair in args.pairs:
        name, _, path = pair.partition("=")
        if not name or not path:
            print(f"author-prompts: '{pair}' is not NAME=TEXT_FILE", file=sys.stderr)
            return 2
        try:
            text = open(path, encoding="utf-8").read()
            if template is None:
                ids, _ = seed_ids(tok, text, seed_tokens, vocab)
            else:
                ids = chat_ids(tok, template, text, seed_tokens, vocab)
        except ValueError as e:
            print(f"author-prompts: {path}: {e}", file=sys.stderr)
            return 1
        out = os.path.join(args.out, f"{name}.tokens.json")
        with open(out, "w") as fp:
            json.dump(ids, fp)
        print(f"author-prompts: wrote {out} ({seed_tokens} tokens)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
