import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "code" / "e0_suite.py"
SPEC = importlib.util.spec_from_file_location("e0_suite", MODULE_PATH)
E0 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(E0)


class E0MathematicalConsistencyTests(unittest.TestCase):
    def test_binary_channel_correction_recovers_conditional_expectation(self):
        fp, fn = 0.2, 0.3
        observed_expectation = np.array([fp, 1.0 - fn])
        corrected = E0.corrected_binary_reward(observed_expectation, fp, fn)
        np.testing.assert_allclose(corrected, np.array([0.0, 1.0]), atol=1e-12)

    def test_singular_channel_is_rejected(self):
        with self.assertRaises(ValueError):
            E0.corrected_binary_reward(np.array([0.0]), 0.4, 0.6)

    def test_preregistered_dr_form_collapses_to_channel_only(self):
        observed = np.array([0.0, 1.0, 1.0, 0.0])
        regression_a = np.array([0.1, 0.2, 0.8, 0.7])
        regression_b = 1.0 - regression_a
        candidate_a = E0.doubly_robust_pseudo_reward(observed, regression_a, 0.15, 0.25)
        candidate_b = E0.doubly_robust_pseudo_reward(observed, regression_b, 0.15, 0.25)
        channel_only = E0.corrected_binary_reward(observed, 0.15, 0.25)
        np.testing.assert_allclose(candidate_a, channel_only, atol=1e-12)
        np.testing.assert_allclose(candidate_b, channel_only, atol=1e-12)

    def test_equal_leverage_scores_collapse_to_uniform(self):
        q = E0._mix_sampling(np.ones(100), 0.2)
        np.testing.assert_allclose(q, np.full(100, 0.01), atol=1e-12)

    def test_hoeffding_width_shrinks_with_audit_size(self):
        widths = [np.sqrt(np.log(20.0) / (2.0 * n)) for n in (50, 100, 200, 500)]
        self.assertTrue(all(a > b for a, b in zip(widths, widths[1:])))

    def test_frozen_suite_has_exactly_six_branches(self):
        self.assertEqual(set(E0.RUNNERS), {"H-001", "H-004", "H-005", "H-008", "H-014", "H-018"})


if __name__ == "__main__":
    unittest.main()
