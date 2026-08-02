import importlib.util
import unittest
from pathlib import Path

import numpy as np
import yaml


MODULE_PATH = Path(__file__).resolve().parents[1] / "code" / "h027_gradient_set_e0.py"
SPEC = importlib.util.spec_from_file_location("h027_gradient_set_e0", MODULE_PATH)
H027 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(H027)


class H027GradientSetConsistencyTests(unittest.TestCase):
    def test_segment_projection_has_closed_form_solution(self):
        minimum, hull = H027.minimum_norm_point([[1.0, -1.0], [1.0, 1.0]])
        np.testing.assert_allclose(minimum, [1.0, 0.0], atol=1e-12)
        self.assertEqual(len(hull), 2)

    def test_origin_in_hull_forces_abstention(self):
        result = H027.maximin_direction([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
        self.assertTrue(result["abstained"])
        np.testing.assert_allclose(result["direction"], [0.0, 0.0], atol=1e-12)

    def test_separated_hull_has_positive_worst_case_margin(self):
        result = H027.maximin_direction([[1.0, -0.5], [1.0, 0.5], [2.0, -0.5], [2.0, 0.5]])
        self.assertFalse(result["abstained"])
        self.assertGreater(result["worst_case_alignment_margin"], 0.0)
        np.testing.assert_allclose(result["direction"], [1.0, 0.0], atol=1e-12)

    def test_valid_channel_interval_contains_clean_gradient(self):
        config = yaml.safe_load(H027.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        clean = np.asarray([0.9, 0.2])
        vertices, _ = H027.audit_compatible_vertices(clean, 0.10, 0.16, 60.0, config)
        hull = H027.convex_hull(vertices)
        self.assertTrue(H027.point_in_convex_hull(clean, hull))

    def test_point_limit_equals_clean_gradient_direction(self):
        config = yaml.safe_load(H027.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        clean = np.asarray([0.9, 0.2])
        vertices, _ = H027.audit_compatible_vertices(clean, 0.0, 0.24, 90.0, config)
        result = H027.maximin_direction(vertices)
        np.testing.assert_allclose(result["direction"], H027.unit(clean), atol=1e-12)

    def test_frozen_grid_matches_preregistration(self):
        config = yaml.safe_load(H027.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(config["seeds"], [11, 23, 47, 89, 131])
        self.assertEqual(config["interval_half_width_grid"], [0.00, 0.02, 0.05, 0.10, 0.20])
        self.assertEqual(config["channel_asymmetry_grid"], [0.00, 0.08, 0.16, 0.24])
        self.assertEqual(config["score_gradient_angle_grid"], [0, 30, 60, 90])
        self.assertEqual(config["success_thresholds"]["minimum_asymmetric_noncollinear_gain_cells"], 3)


if __name__ == "__main__":
    unittest.main()
