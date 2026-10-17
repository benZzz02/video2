"""CPU tests for the present/memory product-of-experts answer."""

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from onestreamer import two_views


class TwoViewTests(unittest.TestCase):
    def test_candidates_by_eval_mode(self):
        mcq = SimpleNamespace(eval_mode="mcq_letter", options={"A": "x", "B": "y", "C": "z"})
        self.assertEqual(list(two_views.candidates(mcq)), ["A", "B", "C"])
        self.assertEqual(set(two_views.candidates(SimpleNamespace(eval_mode="substring_match"))), {"Yes", "No"})
        self.assertEqual(list(two_views.candidates(SimpleNamespace(eval_mode="integer_match"))), list("0123456789"))

    def test_distribution_sums_case_variants_and_renormalizes(self):
        logits = torch.full((10,), float(torch.log(torch.tensor(0.5 / 7))))
        logits[[1, 2, 3]] = torch.log(torch.tensor([0.2, 0.2, 0.1]))
        dist, mass = two_views.distribution(logits, {"Yes": [1, 2], "No": [3]})
        self.assertAlmostEqual(mass, 0.5, places=5)
        self.assertAlmostEqual(dist["Yes"], 0.8, places=5)

    def test_confident_present_resists_confident_memory_only_when_stronger(self):
        present = {"A": 0.9, "B": 0.1}
        self.assertEqual(two_views.combine(present, {"A": 0.3, "B": 0.7})[0], "A")
        self.assertEqual(two_views.combine({"A": 0.5, "B": 0.5}, {"A": 0.3, "B": 0.7})[0], "B")
        self.assertEqual(two_views.combine(present, {"A": 0.02, "B": 0.98})[0], "B")


if __name__ == "__main__":
    unittest.main()
