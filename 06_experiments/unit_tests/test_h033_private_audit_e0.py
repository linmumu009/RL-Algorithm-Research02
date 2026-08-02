import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import yaml


MODULE_PATH = Path(__file__).resolve().parents[1] / "code" / "h033_private_audit_e0.py"
SPEC = importlib.util.spec_from_file_location("h033_private_audit_e0", MODULE_PATH)
H033 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = H033
SPEC.loader.exec_module(H033)


class H033PrivateAuditConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = yaml.safe_load(H033.DEFAULT_CONFIG.read_text(encoding="utf-8"))

    def test_clipping_enforces_per_record_l2_bound(self):
        rows = np.asarray([[3.0, 4.0], [0.3, 0.4], [0.0, 0.0]])
        clipped = H033.clip_rows(rows, 1.0)
        self.assertTrue(np.all(np.linalg.norm(clipped, axis=1) <= 1.0 + 1.0e-12))
        np.testing.assert_allclose(clipped[0], [0.6, 0.8], atol=1.0e-12)
        np.testing.assert_allclose(clipped[1], rows[1], atol=1.0e-12)

    def test_gaussian_calibration_matches_zcdp_formula(self):
        sensitivity = 0.125
        rho = 0.5
        self.assertAlmostEqual(H033.gaussian_sigma(sensitivity, rho), 0.125)

    def test_privacy_filter_never_exceeds_budget(self):
        accountant = H033.RhoAccountant(1.0)
        for _ in range(10):
            self.assertTrue(accountant.can_release(0.1))
            accountant.spend(0.1)
        self.assertFalse(accountant.can_release(0.1))
        self.assertAlmostEqual(accountant.spent, 1.0)

    def test_fast_query_path_equals_explicit_clipped_mean(self):
        table = H033.make_binary_table(11, 64, 8, 91)
        theta = H033.candidate_pool(11, 8, 4)[0]
        signal = H033.signal_vector(8, 0.08)
        fast = H033.clipped_query_mean(table, theta, signal, 0.55, 1.0)
        explicit = H033.clip_rows(H033.query_contributions(table, theta, signal, 0.55), 1.0).mean(axis=0)
        np.testing.assert_allclose(fast, explicit, atol=1.0e-12)

    def test_privacy_off_limit_is_exact_naive_reuse(self):
        config = dict(self.config)
        audit = H033.make_binary_table(23, 64, 4, 92)
        population = H033.make_binary_table(23, 512, 4, 93)
        candidates = H033.candidate_pool(23, 4, 8)
        signal = H033.signal_vector(4, 0.08)
        private_off = H033.run_trace(
            H033.H033_METHOD,
            audit,
            population,
            candidates,
            12,
            1.0,
            signal,
            config,
            23,
            privacy_off=True,
        )
        naive = H033.run_trace(
            "naive_exact_audit_reuse",
            audit,
            population,
            candidates,
            12,
            1.0,
            signal,
            config,
            23,
        )
        self.assertEqual(private_off["candidate_sequence_sha256"], naive["candidate_sequence_sha256"])
        self.assertAlmostEqual(
            private_off["population_gradient_bias_mean"], naive["population_gradient_bias_mean"]
        )
        self.assertAlmostEqual(private_off["clean_gradient_cosine_mean"], naive["clean_gradient_cosine_mean"])

    def test_disjoint_split_uses_each_declared_partition_once(self):
        config = dict(self.config)
        config["disjoint_min_partition_size"] = 16
        audit = H033.make_binary_table(47, 64, 4, 94)
        population = H033.make_binary_table(47, 512, 4, 95)
        candidates = H033.candidate_pool(47, 4, 8)
        signal = H033.signal_vector(4, 0.08)
        trace = H033.run_trace(
            "disjoint_sample_splitting",
            audit,
            population,
            candidates,
            20,
            1.0,
            signal,
            config,
            47,
        )
        self.assertEqual(trace["usable_release_rounds"], 4)

    def test_nonadaptive_sequence_does_not_depend_on_releases(self):
        config = dict(self.config)
        audit = H033.make_binary_table(89, 64, 4, 96)
        population = H033.make_binary_table(89, 512, 4, 97)
        candidates = H033.candidate_pool(89, 4, 8)
        signal = H033.signal_vector(4, 0.08)
        left = H033.run_trace(
            H033.H033_METHOD, audit, population, candidates, 16, 0.1, signal, config, 89, adaptive=False
        )
        right = H033.run_trace(
            H033.H033_METHOD, audit, population, candidates, 16, 3.0, signal, config, 89, adaptive=False
        )
        self.assertEqual(left["candidate_sequence_sha256"], right["candidate_sequence_sha256"])

    def test_frozen_grid_and_comparators_match_preregistration(self):
        self.assertEqual(self.config["seeds"], [11, 23, 47, 89, 131])
        self.assertEqual(self.config["audit_size_grid"], [128, 256, 512])
        self.assertEqual(self.config["adaptive_query_count_grid"], [10, 50, 100, 250])
        self.assertEqual(self.config["zcdp_rho_budget_grid"], [0.10, 0.30, 1.00, 3.00])
        self.assertEqual(self.config["score_dimension_grid"], [2, 8, 32])
        self.assertEqual(self.config["contribution_l2_clip"], 1.0)
        self.assertEqual(len(self.config["baselines"]), 7)
        self.assertEqual(len(self.config["controls"]), 7)
        self.assertEqual(self.config["success_thresholds"]["minimum_adaptive_gain_cells"], 3)

    def test_small_assembly_covers_all_methods_gains_and_controls(self):
        config = dict(self.config)
        config.update(
            {
                "seeds": [11],
                "population_size": 128,
                "query_candidate_count": 8,
                "disjoint_min_partition_size": 16,
                "control_audit_size": 64,
                "control_query_count": 4,
                "control_score_dimension": 4,
            }
        )
        audit = H033.make_binary_table(11, 64, 4, 101)
        population = H033.make_binary_table(11, 128, 4, 102)
        candidates = H033.candidate_pool(11, 4, 8)
        signal = H033.signal_vector(4, 0.08)
        methods = [*config["baselines"], H033.H033_METHOD]
        traces = [
            H033.run_trace(method, audit, population, candidates, 4, 1.0, signal, config, 11)
            for method in methods
        ]
        H033._annotate_gains(traces)
        rows = [
            {
                **H033._row_coordinates(11, 64, 4, 1.0, 4),
                **trace,
            }
            for trace in traces
        ]
        cells = H033._aggregate_cells(rows, config)
        controls = H033._control_rows(config)
        self.assertEqual(len(traces), 8)
        self.assertEqual(len(cells), 8)
        self.assertIn("cosine_gain_over_best_exact_nonoracle", traces[-1])
        self.assertEqual({row["control"] for row in controls}, set(config["controls"]))


if __name__ == "__main__":
    unittest.main()
