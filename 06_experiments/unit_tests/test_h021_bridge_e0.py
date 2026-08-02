import importlib.util
import unittest
from pathlib import Path

import numpy as np
import yaml


MODULE_PATH = Path(__file__).resolve().parents[1] / "code" / "h021_bridge_e0.py"
SPEC = importlib.util.spec_from_file_location("h021_bridge_e0", MODULE_PATH)
H021 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(H021)


class H021BridgeConsistencyTests(unittest.TestCase):
    def test_exact_bridge_recovers_known_solution(self):
        matrix = np.asarray([[0.8, 0.2], [0.2, 0.8]])
        expected = np.asarray([0.15, 0.85])
        target = matrix @ expected
        recovered, condition = H021.solve_bridge(matrix, target, ridge=0.0)
        np.testing.assert_allclose(recovered, expected, atol=1e-12)
        self.assertGreater(condition, 1.0)

    def test_proxy_rank_degrades_toward_random(self):
        conditions = [np.linalg.cond(H021.proxy_transition(value)) for value in (0.90, 0.80, 0.70, 0.55, 0.52)]
        self.assertTrue(all(left < right for left, right in zip(conditions, conditions[1:])))

    def test_rank_collapse_is_finite_under_frozen_ridge(self):
        solution, condition = H021.solve_bridge(H021.proxy_transition(0.50), np.asarray([0.4, 0.6]), ridge=1e-6)
        self.assertTrue(np.all(np.isfinite(solution)))
        self.assertEqual(condition, 1.0e12)

    def test_frozen_grid_matches_preregistration(self):
        config = yaml.safe_load(H021.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(config["seeds"], [11, 23, 47, 89, 131])
        self.assertEqual(config["proxy_relevance_grid"], [0.90, 0.80, 0.70, 0.55, 0.52, 0.50])
        self.assertEqual(config["latent_exploit_strength_grid"], [0.00, 0.15, 0.30])
        self.assertEqual(config["success_thresholds"]["minimum_gain_cells"], 3)


if __name__ == "__main__":
    unittest.main()
