#!/usr/bin/env python3
"""Render user units without systemd, a user bus, or network access."""
import configparser
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest


HERE = Path(__file__).resolve().parent


class ServiceRenderTest(unittest.TestCase):
    def check_render(self, repo, output):
        fake_bin = output.parent / "bin"
        fake_bin.mkdir()
        systemctl = fake_bin / "systemctl"
        systemctl.write_text("#!/bin/sh\necho 'render must not call systemctl' >&2\nexit 125\n")
        systemctl.chmod(0o755)
        env = dict(os.environ, PATH=f"{fake_bin}:{os.environ.get('PATH', '/usr/bin')}")
        command = ["bash", str(repo / "tools/router/install-service.sh"),
                   "--render", str(output)]
        subprocess.run(command, check=True, capture_output=True, text=True, env=env)
        first = {path.name: path.read_bytes() for path in output.glob("*.service")}
        subprocess.run(command, check=True, capture_output=True, text=True, env=env)
        self.assertEqual(first, {path.name: path.read_bytes()
                                 for path in output.glob("*.service")})
        self.assertEqual(set(first), {"nemotron35-router.service", "nemotron35-watch.service"})
        for name, interpreter, script, args in (
            ("router", "/usr/bin/python3", "router.py", ["run"]),
            ("watch", "/usr/bin/bash", "watch.sh", []),
        ):
            with self.subTest(service=name):
                unit = configparser.ConfigParser(interpolation=None)
                unit.read(output / f"nemotron35-{name}.service")
                service = unit["Service"]
                self.assertEqual(service["Restart"], "on-failure")
                self.assertEqual(service["RestartSec"], "30")
                # systemd takes WorkingDirectory= verbatim (no quotes allowed), unlike ExecStart=.
                self.assertEqual(service["WorkingDirectory"], str(repo))
                command = shlex.split(service["ExecStart"])
                self.assertEqual(command[0], "/usr/bin/systemd-inhibit")
                self.assertIn("--what=sleep:idle", command)
                expected = [interpreter]
                if name == "router":
                    expected.append("-u")
                expected += [str(repo / "tools/router" / script), *args]
                self.assertEqual(command[-len(expected):], expected)
                self.assertEqual(service["StandardOutput"], "journal")
                self.assertEqual(service["StandardError"], "journal")
                self.assertEqual(unit["Install"]["WantedBy"], "default.target")

    def test_current_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.check_render(HERE.parent.parent, Path(tmp) / "units")

    def test_relocated_checkout_with_spaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "checkout with spaces"
            shutil.copytree(HERE, repo / "tools/router",
                            ignore=shutil.ignore_patterns("__pycache__"))
            self.check_render(repo, Path(tmp) / "units")


if __name__ == "__main__":
    unittest.main()
