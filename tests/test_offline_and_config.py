"""Step 5b: the whole pipeline inside assert_offline() with zero attempts.
Step 6: no decision module reads a literal threshold."""
from __future__ import annotations

import ast
import socket
import unittest
from pathlib import Path

from winnow import pipeline, telemetry
from winnow.types import NetworkAttempt

PKG = Path(__file__).resolve().parent.parent / "winnow"
DECISION_MODULES = ("classify", "segment", "reading_order", "caption_binding", "speech", "describe",
                    "selfcheck", "pdf_adapter", "pipeline")
# Not thresholds: identities, unit conversions (72 pt/inch, 1000 px² per kpx²),
# and 100 — the data model's definition of text-layer confidence.
ALLOWED = {0, 1, -1, 2, 72, 1000, 100}


class TestOffline(unittest.TestCase):

    def test_pipeline_runs_offline_with_zero_attempts(self):
        before = telemetry.attempts()
        with telemetry.assert_offline():
            doc = pipeline.run("tests/corpus/lecture.pdf", workers=1)
        self.assertTrue(doc.ok)
        self.assertEqual(telemetry.attempts(), before)

    def test_guard_refuses_and_counts(self):
        before = telemetry.attempts()
        with telemetry.assert_offline():
            with self.assertRaises(NetworkAttempt):
                socket.create_connection(("example.com", 80), timeout=1)
        self.assertEqual(telemetry.attempts(), before + 1)

    def test_guard_is_removed_afterwards(self):
        with telemetry.assert_offline():
            pass
        self.assertEqual(socket.getaddrinfo.__module__, "socket")


class TestSourceHygiene(unittest.TestCase):

    def test_no_control_characters_in_source(self):
        """A scripted edit twice turned a regex's \\b into a literal backspace:
        the pattern silently matched nothing and every test still passed."""
        root = PKG.parent
        bad = [f"{p.relative_to(root)}:{n}" for d in ("winnow", "tests", "tools")
               for p in (root / d).rglob("*.py")
               for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
               if any(ord(c) < 32 and c != "\t" for c in line)]
        self.assertEqual(bad, [])


class TestNoLiteralThresholds(unittest.TestCase):

    def test_no_literal_thresholds_outside_types(self):
        bad = []
        for name in DECISION_MODULES:
            tree = ast.parse((PKG / f"{name}.py").read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Compare):
                    for side in [node.left, *node.comparators]:
                        if (isinstance(side, ast.Constant) and isinstance(side.value, (int, float))
                                and not isinstance(side.value, bool) and side.value not in ALLOWED):
                            bad.append(f"{name}.py:{node.lineno} compares with {side.value}")
                if (isinstance(node, ast.Constant) and isinstance(node.value, float)
                        and node.value not in ALLOWED):
                    bad.append(f"{name}.py:{node.lineno} float literal {node.value}")
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
