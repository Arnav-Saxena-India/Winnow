"""Header-qualified table reading (the taxonomy's ``table``), on word boxes."""
from __future__ import annotations

import unittest

from winnow.tables import speak, structure


def _w(text, x0, top, width=None):
    return {"text": text, "x0": x0, "x1": x0 + (width or 9 * len(text)), "top": top, "bottom": top + 18}


def _grid(rows, xs, top=100, pitch=24):
    return [_w(t, x, top + pitch * i) for i, row in enumerate(rows) for t, x in zip(row, xs) if t]


RULES = [(40, 700, 95), (40, 700, 122), (40, 700, 220)]      # booktabs: top, mid, bottom


class TestTables(unittest.TestCase):

    def test_rows_are_named_by_first_cell_and_values_by_header(self):
        words = _grid([["model", "top-1", "top-5"], ["plain-34", "28.54", "10.02"],
                       ["ResNet-34", "25.03", "7.76"]], [50, 300, 500])
        t = structure(words, RULES, [])
        self.assertEqual(t["header"], ["model", "top-1", "top-5"])
        summary, full = speak("Table 3", "Error rates.", t)
        self.assertEqual(summary, "Table 3: Error rates. 2 rows, 3 columns.")
        self.assertIn("plain-34: top-1 28.54, top-5 10.02. ResNet-34: top-1 25.03, top-5 7.76.", full)

    def test_short_rule_makes_a_header_span_its_columns(self):
        """ "BLEU" over "EN-DE" and "EN-FR" (booktabs' cmidrule under it)."""
        words = [_w("Model", 50, 100), _w("BLEU", 360, 100),
                 _w("EN-DE", 300, 124), _w("EN-FR", 450, 124),
                 _w("ByteNet", 50, 160), _w("23.75", 300, 160), _w("24.10", 450, 160),
                 _w("GNMT", 50, 184), _w("24.60", 300, 184), _w("39.92", 450, 184)]
        rules = [(40, 700, 95), (290, 520, 121), (40, 700, 150), (40, 700, 210)]
        t = structure(words, rules, [])
        self.assertEqual(t["header"], ["Model", "BLEU EN-DE", "BLEU EN-FR"])

    def test_grouped_rows_keep_their_values_apart(self):
        """Rows under one label were merged: "4.91 5.01 5.16" (1706.03762, T3)."""
        words = _grid([["group", "N", "PPL"], ["(A)", "1", "5.29"], ["", "4", "5.00"],
                       ["", "16", "4.91"]], [50, 300, 500])
        t = structure(words, RULES[:2] + [(40, 700, 190)], [])
        self.assertEqual([r[2] for r in t["rows"]], ["5.29", "5.00", "4.91"])
        self.assertEqual([r[0] for r in t["rows"]], ["(A)", "(A)", "(A)"])

    def test_label_centred_on_its_group_names_every_row_between_rules(self):
        """ "(A)" sat on its own line, centred on four rows: the first two were
        named "base" and "(A)" was read as an empty row (1706.03762, T3)."""
        words = _grid([["model", "N", "PPL"], ["base", "6", "4.92"]], [50, 300, 500])
        words += _grid([["", "1", "5.29"], ["", "4", "5.00"], ["", "16", "4.91"],
                        ["", "32", "5.01"]], [50, 300, 500], top=160)
        words += [_w("(A)", 50, 196), _w("big", 50, 262), _w("6", 300, 262), _w("4.33", 500, 262)]
        rules = [(40, 700, 95), (40, 700, 122), (40, 700, 150), (40, 700, 255), (40, 700, 290)]
        t = structure(words, rules, [])
        self.assertEqual([r[0] for r in t["rows"]], ["base", "(A)", "(A)", "(A)", "(A)", "big"])
        self.assertEqual([r[2] for r in t["rows"]], ["4.92", "5.29", "5.00", "4.91", "5.01", "4.33"])

    def test_wrapped_cell_joins_its_row(self):
        words = _grid([["parser", "training", "F1"], ["Dyer et al.", "WSJ only,", "91.7"],
                       ["", "discriminative", ""]], [50, 300, 560])
        t = structure(words, RULES, [])
        self.assertEqual(t["rows"], [["Dyer et al.", "WSJ only, discriminative", "91.7"]])

    def test_label_only_row_stays_its_own_row(self):
        """A wrapped label and a group title look alike; merging a title into
        the row above would name the wrong row, so neither is merged."""
        words = _grid([["method", "err"], ["Deep residual", "3.57"], ["networks", ""]], [50, 500])
        t = structure(words, RULES, [])
        self.assertEqual(t["rows"], [["Deep residual", "3.57"], ["networks", ""]])

    def test_a_single_column_is_not_a_table(self):
        words = _grid([["heading"], ["just one"], ["column here"]], [50])
        self.assertIsNone(structure(words, RULES, []))


if __name__ == "__main__":
    unittest.main()
