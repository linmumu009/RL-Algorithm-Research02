import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import yaml


MODULE_PATH = Path(__file__).resolve().parents[1] / "code" / "h039_sign_certificate_e0.py"
SPEC = importlib.util.spec_from_file_location("h039_sign_certificate_e0", MODULE_PATH)
H039 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = H039
SPEC.loader.exec_module(H039)


class H039SignCertificateConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = yaml.safe_load(H039.DEFAULT_CONFIG.read_text(encoding="utf-8"))

    def test_inverse_binary_channel_recovers_clean_expectation(self):
        clean = np.asarray([0.2, 0.8, 0.5])
        contexts = np.asarray([0, 0, 1])
        channels = np.asarray([[0.1, 0.2], [0.15, 0.1]])
        observed = channels[contexts, 0] + (1.0 - channels[contexts, 0] - channels[contexts, 1]) * clean
        np.testing.assert_allclose(H039.inverse_binary_channel(observed, contexts, channels), clean)

    def test_joint_context_corners_preserve_shared_group_centering(self):
        problem = H039.make_problem(11, 2, 128, 0.05, 0.15, 45.0, self.config)
        vertices, corners = H039.advantage_configurations(
            problem["observed_reward"], problem["context_ids"], problem["group_ids"], problem["channel_intervals"]
        )
        intervals = H039.advantage_intervals(vertices)
        self.assertEqual(len(corners), 16)
        self.assertTrue(np.all(problem["clean_advantage"] >= intervals[:, 0] - 1.0e-12))
        self.assertTrue(np.all(problem["clean_advantage"] <= intervals[:, 1] + 1.0e-12))
        for vertex in vertices:
            for group in np.unique(problem["group_ids"]):
                self.assertAlmostEqual(float(np.sum(vertex[problem["group_ids"] == group])), 0.0)

    def test_certificate_uses_worst_case_signed_margin(self):
        intervals = np.asarray([[0.2, 0.5], [-0.7, -0.1], [-0.2, 0.3], [0.0, 0.4]])
        np.testing.assert_allclose(H039.certified_weights(intervals), [0.2, -0.1, 0.0, 0.0])

    def test_point_identified_limit_matches_h001(self):
        problem = H039.make_problem(23, 2, 128, 0.0, 0.15, 45.0, self.config)
        evaluation = H039.evaluate_problem(problem, 23, self.config)
        self.assertLessEqual(evaluation["point_limit_direction_difference"], 1.0e-12)
        h039 = evaluation["method_metrics"][H039.H039_METHOD]
        h001 = evaluation["method_metrics"]["H001_point_channel_correction"]
        np.testing.assert_allclose(h039["selected_gradient"], h001["selected_gradient"], atol=1.0e-12)

    def test_masks_and_random_filter_match_h039_acceptance_count(self):
        problem = H039.make_problem(47, 4, 64, 0.10, 0.30, 90.0, self.config)
        evaluation = H039.evaluate_problem(problem, 47, self.config)
        self.assertTrue(evaluation["h010_acceptance_matches"])
        self.assertTrue(evaluation["random_acceptance_matches"])

    def test_parameter_and_channel_radius_are_exactly_matched(self):
        problem = H039.make_problem(89, 2, 256, 0.05, 0.15, 45.0, self.config)
        evaluation = H039.evaluate_problem(problem, 89, self.config)
        self.assertAlmostEqual(evaluation["parameter_radius"], evaluation["channel_radius_mean"])

    def test_zero_certificate_control_has_zero_mass(self):
        problem = H039.make_problem(131, 2, 128, 0.05, 0.15, 45.0, self.config)
        evaluation = H039.evaluate_problem(problem, 131, self.config, force_zero_certificate=True)
        self.assertEqual(evaluation["method_metrics"][H039.H039_METHOD]["certified_mass"], 0.0)

    def test_frozen_grid_and_comparators_match_preregistration(self):
        self.assertEqual(self.config["seeds"], [11, 23, 47, 89, 131])
        self.assertEqual(self.config["context_count_grid"], [2, 4])
        self.assertEqual(self.config["audit_size_grid"], [64, 128, 256])
        self.assertEqual(self.config["channel_interval_half_width_grid"], [0.02, 0.05, 0.10, 0.20])
        self.assertEqual(self.config["channel_heterogeneity_grid"], [0.00, 0.15, 0.30])
        self.assertEqual(self.config["score_gradient_noncollinearity_angle_grid"], [0, 45, 90])
        cell_count = 2 * 3 * 4 * 3 * 3
        self.assertEqual(cell_count, 216)
        self.assertEqual(cell_count * len(self.config["seeds"]), 1080)
        self.assertEqual(len(self.config["baselines"]), 8)
        self.assertEqual(len(self.config["controls"]), 8)
        self.assertEqual(self.config["success_thresholds"]["minimum_heterogeneous_noncollinear_gain_cells"], 3)

    def test_small_assembly_contains_all_methods_and_controls(self):
        problem = H039.make_problem(11, 2, 64, 0.02, 0.15, 45.0, self.config)
        evaluation = H039.evaluate_problem(problem, 11, self.config)
        expected_methods = {H039.H039_METHOD, *self.config["baselines"]}
        self.assertEqual(set(evaluation["method_metrics"]), expected_methods)
        controls = H039._control_rows({**self.config, "seeds": [11]})
        self.assertEqual({row["control"] for row in controls}, set(self.config["controls"]))
        self.assertEqual(len(controls), 8)


if __name__ == "__main__":
    unittest.main()
