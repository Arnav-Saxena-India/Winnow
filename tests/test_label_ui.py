"""The labelling window: bookkeeping, and one real window driven by its methods."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from winnow import label_ui

CORPUS = Path(__file__).parent / "corpus"


def _lab():
    return {"doc": "x.pdf", "labeller": "t", "pages": {
        "2": [{"box": [0, 0, 5, 5], "kind": "", "text": "c"}],
        "1": [{"box": [0, 0, 5, 5], "kind": "body", "text": "a"},
              {"box": [0, 9, 5, 14], "kind": "", "text": "b"}]}}


class TestBookkeeping(unittest.TestCase):

    def test_next_open_skips_labelled_boxes_and_wraps(self):
        lab = _lab()
        self.assertEqual(label_ui.next_open(lab), ("1", 1))
        self.assertEqual(label_ui.next_open(lab, ("1", 1)), ("2", 0))
        self.assertEqual(label_ui.next_open(lab, ("2", 0)), ("1", 1))
        lab["pages"]["1"][1]["kind"] = lab["pages"]["2"][0]["kind"] = "chrome"
        self.assertIsNone(label_ui.next_open(lab))

    def test_progress_counts_only_valid_kinds(self):
        lab = _lab()
        lab["pages"]["2"][0]["kind"] = "not-a-kind"
        self.assertEqual(label_ui.progress(lab), (1, 3))

    def test_save_round_trips_and_leaves_no_temp_file(self):
        d = Path(tempfile.mkdtemp())
        label_ui.save(_lab(), d / "a.labels.json")
        self.assertEqual(json.loads((d / "a.labels.json").read_text(encoding="utf-8")), _lab())
        self.assertEqual([p.name for p in d.iterdir()], ["a.labels.json"])


class TestWindow(unittest.TestCase):

    def test_keys_label_boxes_and_drag_adds_one(self):
        import tkinter as tk
        try:
            root = tk.Tk()
        except tk.TclError:
            self.skipTest("no display")
        root.withdraw()
        lab = json.loads((CORPUS / "lecture.labels.json").read_text(encoding="utf-8"))
        lab["doc"] = str((CORPUS / "lecture.pdf").resolve())
        for boxes in lab["pages"].values():
            for b in boxes:
                b["kind"] = ""
        path = Path(tempfile.mkdtemp()) / "lecture.labels.json"
        label_ui.save(lab, path)
        try:
            w = label_ui.Labeller(root, str(path))
            self.assertEqual(w.cur, ("1", 0))
            w.set_kind("chrome")
            self.assertEqual(w.cur, ("1", 1))
            n = len(w.lab["pages"]["1"])

            class E:                       # a mouse event at canvas coordinates
                def __init__(self, x, y):
                    self.x, self.y = x, y
            w._press(E(10, 10))
            w._motion(E(200, 60))
            w._release(E(200, 60))
            w.delete()                     # the added box: not a region after all
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["pages"]["1"][0]["kind"], "chrome")
            self.assertEqual(len(saved["pages"]["1"]), n)
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
