#!/usr/bin/env python3
"""Tests for tools/stitch-prompts.py. Synthetic corpus, a whitespace stand-in
for the pinned tokenizer, a seeded RNG (the CLI itself always uses
/dev/urandom). No tokenizers package, no network."""
from __future__ import annotations

import importlib.util
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("stitch_prompts", HERE / "stitch-prompts.py")
sp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sp)

SEED = 300  # "tokens" = words here


def fake_trim(text):
    words = list(__import__("re").finditer(r"\S+", text))
    if len(words) < SEED:
        raise ValueError("short")
    return [hash(m.group()) % 1000 for m in words[:SEED]], words[SEED - 1].end()


def book(work: str, n: int) -> str:
    paras = ["CHAPTER I."] + [" ".join(f"{work}p{i}w{j}" for j in range(50 + (i * 7) % 40)) for i in range(n)]
    return "Gutenberg header\n*** START OF THE PROJECT\n\n" + "\n\n".join(paras) + "\n\n*** END OF THE PROJECT\nlicense\n"


def corpus(n_works=6, n_paras=30):
    return {f"pg{w}": sp.paragraphs(book(f"pg{w}", n_paras)) for w in range(n_works)}


class StitchTest(unittest.TestCase):
    def test_paragraphs_strip_gutenberg_header_and_footer(self):
        ps = sp.paragraphs(book("pgX", 3))
        self.assertEqual(ps[0], "CHAPTER I.")
        self.assertFalse(any("Gutenberg" in p or "license" in p for p in ps))
        with self.assertRaises(ValueError):
            sp.paragraphs("no markers here")

    def test_prompts_span_four_works_and_never_share_a_paragraph(self):
        works = corpus()
        used = set()
        prompts = sp.stitch(works, "g1", 8, used, random.Random(3), fake_trim)
        self.assertEqual([p[0] for p in prompts], [f"g1-{n}" for n in range(1, 9)])
        seen = set()
        for name, ids, text, pieces in prompts:
            self.assertEqual(len(ids), SEED)
            self.assertGreaterEqual(len({p["work"] for p in pieces}), 4, name)
            for p in pieces:  # provenance points at the real paragraph, which is in the text
                para = works[p["work"]][p["paragraph"]]
                self.assertEqual(p["sha256"], sp.para_key(para))
                self.assertTrue(p["truncated"] or para in text)
                self.assertNotIn(p["sha256"], seen)
                seen.add(p["sha256"])
        self.assertEqual(used, seen)

    def test_deterministic_under_a_seeded_rng(self):
        a = sp.stitch(corpus(), "g1", 3, set(), random.Random(11), fake_trim)
        b = sp.stitch(corpus(), "g1", 3, set(), random.Random(11), fake_trim)
        c = sp.stitch(corpus(), "g1", 3, set(), random.Random(12), fake_trim)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_used_paragraphs_are_never_drawn(self):
        works = corpus()
        final = sp.stitch(works, "final", 4, set(), random.Random(1), fake_trim)
        spent = {p["sha256"] for _, _, _, ps in final for p in ps}
        used = set(spent)
        later = sp.stitch(works, "g2", 4, used, random.Random(2), fake_trim)
        self.assertFalse(spent & {p["sha256"] for _, _, _, ps in later for p in ps})
        # every paragraph of all but 3 works spent: fewer than 4 works left
        used = {sp.para_key(p) for w in list(works)[3:] for p in works[w]}
        with self.assertRaisesRegex(ValueError, "only 3 works"):
            sp.stitch(works, "g3", 1, used, random.Random(1), fake_trim)


class CliTest(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(HERE / "stitch-prompts.py"), "--corpus", "x",
                               "--snapshot", "x", *args], capture_output=True, text=True)

    def test_label_already_in_ledger_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            ledger = Path(t) / "ledger.json"
            ledger.write_text(json.dumps({"labels": ["g1", "final"], "paragraphs": {}}))
            for label in ("g1", "final"):
                r = self.run_cli("--label", label, "--out", f"{t}/out", "--ledger", str(ledger))
                self.assertEqual(r.returncode, 1)
                self.assertIn("already in the ledger", r.stderr)

    def test_out_inside_the_repo_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            r = self.run_cli("--label", "g1", "--out", str(HERE / "leak"), "--ledger", f"{t}/l.json")
            self.assertEqual(r.returncode, 1)
            self.assertIn("inside the repository", r.stderr)
            self.assertFalse((HERE / "leak").exists())

    def test_bad_label_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            r = self.run_cli("--label", "G1", "--out", f"{t}/o", "--ledger", f"{t}/l.json")
            self.assertIn("is not g<N> or final", r.stderr)


if __name__ == "__main__":
    unittest.main()
