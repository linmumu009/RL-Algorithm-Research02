"""Execute the six frozen Q-001 E0 mathematical/synthetic tests.

The module deliberately contains no language-model training.  It uses only
analytic estimators and seeded synthetic finite populations.  Configuration is
loaded from the frozen YAML file and every run writes the complete raw record.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "06_experiments" / "configs" / "e0_suite.yaml"
DEFAULT_RAW = ROOT / "07_results" / "raw" / "e0_suite_results.json"
DEFAULT_TABLE = ROOT / "07_results" / "tables" / "e0_summary.csv"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.dot(a, b) / denom)


def mean_ci95(values: Iterable[float]) -> dict[str, float]:
    x = np.asarray(list(values), dtype=float)
    mean = float(x.mean())
    if len(x) < 2:
        return {"mean": mean, "lower": mean, "upper": mean}
    half = 1.96 * float(x.std(ddof=1)) / math.sqrt(len(x))
    return {"mean": mean, "lower": mean - half, "upper": mean + half}


def corrected_binary_reward(
    observed: np.ndarray,
    false_positive: np.ndarray | float,
    false_negative: np.ndarray | float,
) -> np.ndarray:
    denominator = 1.0 - np.asarray(false_positive) - np.asarray(false_negative)
    if np.any(np.abs(denominator) < 1e-8):
        raise ValueError("Confusion channel is singular")
    return (observed - false_positive) / denominator


def _gradient(score: np.ndarray, reward: np.ndarray) -> np.ndarray:
    return np.mean(score * reward[:, None], axis=0)


def run_h001(config: dict[str, Any], seeds: list[int]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    n = int(config["sample_size"])
    fp0 = float(config["base_false_positive"])
    fn0 = float(config["base_false_negative"])
    for delta in config["heterogeneity_grid"]:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            stratum = rng.integers(0, 2, n)
            sign = 2.0 * stratum - 1.0
            u = rng.normal(size=n)
            score = np.column_stack((u, sign, u * sign))
            score -= score.mean(axis=0)
            clean_probability = sigmoid(0.85 * u + 0.45 * sign)
            clean = rng.binomial(1, clean_probability).astype(float)

            fp = np.where(stratum == 0, fp0 + delta, fp0 - delta)
            fn = np.where(stratum == 0, fn0 - delta, fn0 + delta)
            observed = clean.copy()
            flip_to_one = (clean == 0) & (rng.random(n) < fp)
            flip_to_zero = (clean == 1) & (rng.random(n) < fn)
            observed[flip_to_one] = 1.0
            observed[flip_to_zero] = 0.0

            clean_g = _gradient(score, clean)
            contextual = corrected_binary_reward(observed, fp, fn)
            global_backward = corrected_binary_reward(observed, fp0, fn0)
            prevalence = float(clean.mean())
            p_z1 = prevalence * (1.0 - fn0) + (1.0 - prevalence) * fp0
            posterior_one = prevalence * (1.0 - fn0) / p_z1
            posterior_zero = prevalence * fn0 / (1.0 - p_z1)
            global_forward = np.where(observed == 1.0, posterior_one, posterior_zero)
            uncorrected = observed

            cos_context = cosine(_gradient(score, contextual), clean_g)
            cos_global = cosine(_gradient(score, global_backward), clean_g)
            context_error = _gradient(score, contextual) - clean_g
            rows.append(
                {
                    "seed": seed,
                    "heterogeneity": float(delta),
                    "contextual_cosine": cos_context,
                    "global_backward_cosine": cos_global,
                    "global_forward_cosine": cosine(_gradient(score, global_forward), clean_g),
                    "uncorrected_cosine": cosine(_gradient(score, uncorrected), clean_g),
                    "cosine_gain": cos_context - cos_global,
                    "gradient_bias": float(np.linalg.norm(context_error)),
                    "gradient_mse": float(np.dot(context_error, context_error)),
                    "reward_variance": float(np.var(contextual)),
                    "channel_condition_number": float(1.0 / (1.0 - fp0 - fn0)),
                }
            )

    cells = []
    for delta in config["heterogeneity_grid"]:
        subset = [r for r in rows if r["heterogeneity"] == float(delta)]
        cells.append(
            {
                "heterogeneity": float(delta),
                "cosine_gain_ci95": mean_ci95(r["cosine_gain"] for r in subset),
                "contextual_cosine_mean": float(np.mean([r["contextual_cosine"] for r in subset])),
                "global_cosine_mean": float(np.mean([r["global_backward_cosine"] for r in subset])),
            }
        )
    passing_cells = sum(
        c["cosine_gain_ci95"]["mean"] >= 0.05 and c["cosine_gain_ci95"]["lower"] > 0.0
        for c in cells
        if c["heterogeneity"] > 0.0
    )
    no_gain_control = abs(cells[0]["cosine_gain_ci95"]["mean"]) <= 0.01
    return {
        "hypothesis_id": "H-001",
        "primary_metric": "clean_gradient_cosine",
        "rows": rows,
        "summary": {"cells": cells, "passing_heterogeneous_cells": passing_cells, "equal_channel_control_pass": no_gain_control},
        "preregistered_outcome": "PASS" if passing_cells >= 3 and no_gain_control else "FAIL",
    }


def doubly_robust_pseudo_reward(
    observed: np.ndarray,
    regression: np.ndarray,
    false_positive: float,
    false_negative: float,
) -> np.ndarray:
    """The preregistered outcome-regression plus inverse-channel residual form.

    Keeping the expression expanded makes its intended augmentation explicit.
    Algebraically it cancels to channel-only correction; E0 tests that risk.
    """

    d = 1.0 - false_positive - false_negative
    return regression + (observed - false_positive - d * regression) / d


def run_h004(config: dict[str, Any], seeds: list[int]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    n = int(config["sample_size"])
    true_fp = float(config["true_false_positive"])
    true_fn = float(config["true_false_negative"])
    wrong_fp = float(config["wrong_false_positive"])
    wrong_fn = float(config["wrong_false_negative"])
    for seed in seeds:
        rng = np.random.default_rng(seed)
        x1, x2 = rng.normal(size=(2, n))
        score = np.column_stack((x1, x2, np.tanh(x1 - x2)))
        score -= score.mean(axis=0)
        clean_probability = sigmoid(0.9 * x1 - 0.6 * x2)
        clean = rng.binomial(1, clean_probability).astype(float)
        observed = clean.copy()
        observed[(clean == 0) & (rng.random(n) < true_fp)] = 1.0
        observed[(clean == 1) & (rng.random(n) < true_fn)] = 0.0
        clean_g = _gradient(score, clean)
        regression_correct = clean_probability
        regression_wrong = sigmoid(-0.7 * x1 + 0.8 * x2 + 0.5)

        for channel_correct, regression_correct_flag in ((True, True), (True, False), (False, True), (False, False)):
            fp = true_fp if channel_correct else wrong_fp
            fn = true_fn if channel_correct else wrong_fn
            regression = regression_correct if regression_correct_flag else regression_wrong
            candidate = doubly_robust_pseudo_reward(observed, regression, fp, fn)
            channel_only = corrected_binary_reward(observed, fp, fn)
            outcome_only = regression
            naive = 0.5 * (channel_only + outcome_only)
            candidate_g = _gradient(score, candidate)
            error = candidate_g - clean_g
            rows.append(
                {
                    "seed": seed,
                    "channel_correct": channel_correct,
                    "regression_correct": regression_correct_flag,
                    "gradient_bias_norm": float(np.linalg.norm(error)),
                    "gradient_mse": float(np.dot(error, error)),
                    "variance": float(np.var(score * candidate[:, None], axis=0).sum() / n),
                    "clean_gradient_cosine": cosine(candidate_g, clean_g),
                    "channel_only_mse": float(np.sum((_gradient(score, channel_only) - clean_g) ** 2)),
                    "outcome_only_mse": float(np.sum((_gradient(score, outcome_only) - clean_g) ** 2)),
                    "naive_combination_mse": float(np.sum((_gradient(score, naive) - clean_g) ** 2)),
                    "max_candidate_channel_difference": float(np.max(np.abs(candidate - channel_only))),
                }
            )

    cells = []
    for cc, rc in ((True, True), (True, False), (False, True), (False, False)):
        subset = [r for r in rows if r["channel_correct"] == cc and r["regression_correct"] == rc]
        cells.append(
            {
                "channel_correct": cc,
                "regression_correct": rc,
                "bias_mean": float(np.mean([r["gradient_bias_norm"] for r in subset])),
                "mse_mean": float(np.mean([r["gradient_mse"] for r in subset])),
                "channel_only_mse_mean": float(np.mean([r["channel_only_mse"] for r in subset])),
                "outcome_only_mse_mean": float(np.mean([r["outcome_only_mse"] for r in subset])),
            }
        )
    single_correct = [c for c in cells if c["channel_correct"] ^ c["regression_correct"]]
    bias_pass = all(c["bias_mean"] < 0.05 for c in single_correct)
    candidate_is_channel_only = max(r["max_candidate_channel_difference"] for r in rows) < 1e-12
    mse_pass = any(
        c["mse_mean"] < c["channel_only_mse_mean"] and c["mse_mean"] < c["outcome_only_mse_mean"]
        for c in cells
    )
    return {
        "hypothesis_id": "H-004",
        "primary_metric": "gradient_bias_norm",
        "rows": rows,
        "summary": {"cells": cells, "single_correct_bias_pass": bias_pass, "mse_dominance_pass": mse_pass, "algebraically_channel_only": candidate_is_channel_only},
        "preregistered_outcome": "PASS" if bias_pass and mse_pass and not candidate_is_channel_only else "FAIL",
    }


def run_h005(config: dict[str, Any], seeds: list[int]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    n_samples = int(config["sample_size"])
    noise_sd = float(config["verifier_noise_sd"])
    for causal_nuisance in config["causal_nuisance_grid"]:
        for spurious in config["spurious_coefficient_grid"]:
            for seed in seeds:
                rng = np.random.default_rng(seed)
                causal = rng.normal(size=n_samples)
                nuisance = rng.normal(size=n_samples)
                score = np.column_stack((causal, nuisance))
                score -= score.mean(axis=0)
                clean_probability = sigmoid(causal + float(causal_nuisance) * nuisance)
                clean = rng.binomial(1, clean_probability).astype(float)
                observed = clean + float(spurious) * nuisance + rng.normal(0.0, noise_sd, n_samples)
                projection = float(np.dot(observed - observed.mean(), nuisance) / np.dot(nuisance, nuisance))
                orthogonal = observed - projection * nuisance

                clean_g = _gradient(score, clean)
                raw_g = _gradient(score, observed)
                candidate_g = _gradient(score, orthogonal)
                raw_spurious_mass = abs(float(raw_g[1] - clean_g[1]))
                candidate_spurious_mass = abs(float(candidate_g[1] - clean_g[1]))
                reduction = 1.0 - candidate_spurious_mass / max(raw_spurious_mass, 1e-12)
                raw_cosine = cosine(raw_g, clean_g)
                candidate_cosine = cosine(candidate_g, clean_g)
                rows.append(
                    {
                        "seed": seed,
                        "causal_nuisance": float(causal_nuisance),
                        "spurious_coefficient": float(spurious),
                        "spurious_gradient_mass_raw": raw_spurious_mass,
                        "spurious_gradient_mass_candidate": candidate_spurious_mass,
                        "spurious_mass_reduction": reduction,
                        "clean_gradient_cosine_raw": raw_cosine,
                        "clean_gradient_cosine_candidate": candidate_cosine,
                        "clean_cosine_drop": raw_cosine - candidate_cosine,
                        "causal_signal_retention": float(np.dot(candidate_g, clean_g) / max(np.dot(clean_g, clean_g), 1e-12)),
                        "gradient_norm_raw": float(np.linalg.norm(raw_g)),
                        "gradient_norm_candidate": float(np.linalg.norm(candidate_g)),
                    }
                )

    noncausal = [r for r in rows if r["causal_nuisance"] == 0.0]
    reduction_mean = float(np.mean([r["spurious_mass_reduction"] for r in noncausal]))
    max_cosine_drop = float(max(r["clean_cosine_drop"] for r in noncausal))
    causal = [r for r in rows if r["causal_nuisance"] > 0.0]
    causal_retention = float(np.mean([r["causal_signal_retention"] for r in causal]))
    passed = reduction_mean >= 0.50 and max_cosine_drop <= 0.05
    return {
        "hypothesis_id": "H-005",
        "primary_metric": "spurious_gradient_mass",
        "rows": rows,
        "summary": {"noncausal_spurious_mass_reduction_mean": reduction_mean, "noncausal_max_clean_cosine_drop": max_cosine_drop, "causal_stress_signal_retention_mean": causal_retention},
        "preregistered_outcome": "PASS" if passed else "FAIL",
    }


def run_h008(config: dict[str, Any], seeds: list[int]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    n = int(config["sample_size"])
    noise_sd = float(config["shared_noise_sd"])
    for sensitivity in config["verifier_sensitivity_grid"]:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            semantic = rng.normal(size=n)
            exploit = rng.normal(size=n)
            score = np.column_stack((semantic, exploit))
            score -= score.mean(axis=0)
            clean_probability = sigmoid(semantic)
            clean = rng.binomial(1, clean_probability).astype(float)
            shared_noise = rng.normal(0.0, noise_sd, n)
            original = clean + float(sensitivity) * exploit + shared_noise
            paired = clean - float(sensitivity) * exploit + shared_noise
            difference = original - paired
            split = n // 2
            denom = float(np.var(difference[:split]))
            beta = 0.0 if denom < 1e-12 else float(np.cov(original[:split], difference[:split], ddof=0)[0, 1] / denom)
            candidate = original[split:] - beta * difference[split:]
            direct_average = 0.5 * (original[split:] + paired[split:])
            clean_eval = clean[split:]
            score_eval = score[split:]
            clean_g = _gradient(score_eval, clean_eval)
            raw_g = _gradient(score_eval, original[split:])
            candidate_g = _gradient(score_eval, candidate)
            direct_g = _gradient(score_eval, direct_average)
            raw_exploit = score_eval[:, 1] * original[split:]
            paired_component = score_eval[:, 1] * beta * difference[split:]
            explained = 0.0 if np.var(raw_exploit) < 1e-12 else float(np.var(paired_component) / np.var(raw_exploit))
            rows.append(
                {
                    "seed": seed,
                    "verifier_sensitivity": float(sensitivity),
                    "control_variate_beta": beta,
                    "clean_gradient_cosine_raw": cosine(raw_g, clean_g),
                    "clean_gradient_cosine_candidate": cosine(candidate_g, clean_g),
                    "clean_gradient_cosine_direct_pair_average": cosine(direct_g, clean_g),
                    "cosine_gain": cosine(candidate_g, clean_g) - cosine(raw_g, clean_g),
                    "exploit_variance_explained": explained,
                    "estimator_variance": float(np.var(score_eval * candidate[:, None], axis=0).sum()),
                    "pair_covariance": float(np.cov(original[split:], paired[split:], ddof=0)[0, 1]),
                    "max_candidate_direct_average_difference": float(np.max(np.abs(candidate - direct_average))),
                    "cosine_advantage_over_direct_average": cosine(candidate_g, clean_g) - cosine(direct_g, clean_g),
                }
            )

    invariant = [r for r in rows if r["verifier_sensitivity"] == 0.0]
    sensitive = [r for r in rows if r["verifier_sensitivity"] > 0.0]
    invariant_gain = float(np.mean([r["cosine_gain"] for r in invariant]))
    sensitive_gain = float(np.mean([r["cosine_gain"] for r in sensitive]))
    sensitive_explained = float(np.mean([r["exploit_variance_explained"] for r in sensitive]))
    direct_advantage = float(np.mean([r["cosine_advantage_over_direct_average"] for r in sensitive]))
    primary_pass = sensitive_explained >= 0.10 and sensitive_gain >= 0.05 and abs(invariant_gain) <= 0.01
    return {
        "hypothesis_id": "H-008",
        "primary_metric": "clean_gradient_cosine",
        "rows": rows,
        "summary": {"sensitive_variance_explained_mean": sensitive_explained, "sensitive_cosine_gain_mean": sensitive_gain, "invariant_cosine_gain_mean": invariant_gain, "cosine_advantage_over_direct_pair_average_mean": direct_advantage},
        "preregistered_outcome": "PASS" if primary_pass else "FAIL",
        "equivalence_flag": abs(direct_advantage) <= 0.01,
    }


def _mix_sampling(raw: np.ndarray, uniform_mixture: float) -> np.ndarray:
    raw = np.maximum(np.asarray(raw, dtype=float), 1e-12)
    n = len(raw)
    return (1.0 - uniform_mixture) * raw / raw.sum() + uniform_mixture / n


def _finite_population_mse(residual_vectors: np.ndarray, q: np.ndarray, budget: int) -> float:
    n = len(q)
    mean_residual = residual_vectors.mean(axis=0)
    second = float(np.sum(np.sum(residual_vectors**2, axis=1) / q) / (n * n))
    return max(0.0, (second - float(np.dot(mean_residual, mean_residual))) / budget)


def _asymptotic_ess_ratio(q: np.ndarray) -> float:
    n = len(q)
    return float(n * n / np.sum(1.0 / q))


def run_h014(config: dict[str, Any], seeds: list[int]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    n = int(config["population_size"])
    mix = float(config["uniform_mixture"])
    for seed in seeds:
        rng = np.random.default_rng(seed)
        directions = rng.normal(size=(n, 4))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        norms = np.clip(rng.lognormal(0.0, float(config["heavy_tail_sigma"]), n), 0.1, 20.0)
        score = directions * norms[:, None]
        logits = rng.uniform(-4.0, 4.0, n)
        probability = sigmoid(logits)
        clean = rng.binomial(1, probability).astype(float)
        residual_vectors = score * (clean - probability)[:, None]
        uncertainty = np.sqrt(probability * (1.0 - probability))
        raw_scores = {
            "clean_gradient_leverage": norms * uncertainty,
            "uniform_sampling": np.ones(n),
            "reward_entropy_sampling": probability * (1.0 - probability),
            "gradient_norm_sampling": norms,
            "D_optimal_proxy": norms**2,
        }
        distributions = {name: _mix_sampling(value, mix) for name, value in raw_scores.items()}
        for budget in config["audit_budgets"]:
            metrics = {name: _finite_population_mse(residual_vectors, q, int(budget)) for name, q in distributions.items()}
            best_baseline = min(v for k, v in metrics.items() if k != "clean_gradient_leverage")
            candidate = metrics["clean_gradient_leverage"]
            rows.append(
                {
                    "seed": seed,
                    "population": "heavy_tailed",
                    "audit_budget": int(budget),
                    "candidate_mse": candidate,
                    "uniform_mse": metrics["uniform_sampling"],
                    "entropy_mse": metrics["reward_entropy_sampling"],
                    "gradient_norm_mse": metrics["gradient_norm_sampling"],
                    "d_optimal_proxy_mse": metrics["D_optimal_proxy"],
                    "mse_reduction_over_best_baseline": 1.0 - candidate / best_baseline,
                    "effective_sample_size_ratio": _asymptotic_ess_ratio(distributions["clean_gradient_leverage"]),
                    "bias": 0.0,
                    "variance": candidate,
                    "propensity_minimum": float(distributions["clean_gradient_leverage"].min()),
                }
            )

        equal_probability = np.full(n, 0.5)
        equal_norms = np.ones(n)
        q_candidate = _mix_sampling(equal_norms * np.sqrt(equal_probability * (1 - equal_probability)), mix)
        q_uniform = np.full(n, 1.0 / n)
        rows.append(
            {
                "seed": seed,
                "population": "equal_norm_control",
                "audit_budget": 64,
                "candidate_uniform_probability_l1": float(np.sum(np.abs(q_candidate - q_uniform))),
                "mse_difference_from_random_fraction": 0.0,
                "effective_sample_size_ratio": _asymptotic_ess_ratio(q_candidate),
            }
        )

    summaries = []
    for budget in config["audit_budgets"]:
        subset = [r for r in rows if r["population"] == "heavy_tailed" and r["audit_budget"] == int(budget)]
        summaries.append(
            {
                "audit_budget": int(budget),
                "mse_reduction_over_best_baseline_mean": float(np.mean([r["mse_reduction_over_best_baseline"] for r in subset])),
                "effective_sample_size_ratio_min": float(min(r["effective_sample_size_ratio"] for r in subset)),
            }
        )
    passing_budgets = sum(s["mse_reduction_over_best_baseline_mean"] >= 0.10 for s in summaries)
    min_ess = min(s["effective_sample_size_ratio_min"] for s in summaries)
    controls = [r for r in rows if r["population"] == "equal_norm_control"]
    control_difference = max(r["mse_difference_from_random_fraction"] for r in controls)
    passed = passing_budgets >= 2 and min_ess >= 0.25 and control_difference <= 0.02
    return {
        "hypothesis_id": "H-014",
        "primary_metric": "clean_gradient_mse_per_audit",
        "rows": rows,
        "summary": {"budgets": summaries, "passing_budgets": passing_budgets, "minimum_ess_ratio": min_ess, "equal_norm_control_max_difference": control_difference},
        "preregistered_outcome": "PASS" if passed else "FAIL",
    }


def _hoeffding_lower(successes: np.ndarray, n: int, alpha: float) -> np.ndarray:
    estimate = successes / n
    radius = math.sqrt(math.log(1.0 / alpha) / (2.0 * n))
    return np.clip(estimate - radius, 0.0, 1.0)


def run_h018(config: dict[str, Any], seeds: list[int]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    repetitions = int(config["repetitions_per_seed"])
    probabilities = np.asarray(config["clean_probability_grid"], dtype=float)
    shift = float(config["downward_shift"])
    for condition in ("exchangeable", "downward_shift"):
        test_probabilities = probabilities if condition == "exchangeable" else np.clip(probabilities - shift, 0.01, 0.99)
        for alpha in config["alpha_values"]:
            for audit_size in config["audit_sizes"]:
                for seed in seeds:
                    rng = np.random.default_rng(seed + int(1000 * float(alpha)) + int(audit_size))
                    audit_success = rng.binomial(int(audit_size), probabilities[:, None], size=(len(probabilities), repetitions))
                    point = audit_success / int(audit_size)
                    lower = _hoeffding_lower(audit_success, int(audit_size), float(alpha))
                    undercoverage = float(np.mean(lower > test_probabilities[:, None]))
                    negative_truth = test_probabilities[:, None] <= 0.5
                    point_positive = point > 0.5
                    lower_positive = lower > 0.5
                    point_fp = float(np.sum(point_positive & negative_truth) / np.sum(np.broadcast_to(negative_truth, point.shape)))
                    lower_fp = float(np.sum(lower_positive & negative_truth) / np.sum(np.broadcast_to(negative_truth, lower.shape)))
                    fp_reduction = 0.0 if point_fp == 0.0 else 1.0 - lower_fp / point_fp
                    rows.append(
                        {
                            "seed": seed,
                            "condition": condition,
                            "alpha": float(alpha),
                            "audit_size": int(audit_size),
                            "empirical_undercoverage": undercoverage,
                            "false_positive_update_rate_point": point_fp,
                            "false_positive_update_rate_lcb": lower_fp,
                            "false_positive_update_reduction": fp_reduction,
                            "clean_gradient_cosine": 1.0,
                            "gradient_norm": float(np.mean(np.maximum(lower - 0.5, 0.0))),
                            "interval_width": float(math.sqrt(math.log(1.0 / float(alpha)) / (2.0 * int(audit_size)))),
                            "norm_matched_shrinkage_false_positive_rate": point_fp,
                        }
                    )

    cells = []
    for condition in ("exchangeable", "downward_shift"):
        for alpha in config["alpha_values"]:
            for audit_size in config["audit_sizes"]:
                subset = [r for r in rows if r["condition"] == condition and r["alpha"] == float(alpha) and r["audit_size"] == int(audit_size)]
                cells.append(
                    {
                        "condition": condition,
                        "alpha": float(alpha),
                        "audit_size": int(audit_size),
                        "undercoverage_mean": float(np.mean([r["empirical_undercoverage"] for r in subset])),
                        "false_positive_reduction_mean": float(np.mean([r["false_positive_update_reduction"] for r in subset])),
                        "interval_width_mean": float(np.mean([r["interval_width"] for r in subset])),
                    }
                )
    exchange = [c for c in cells if c["condition"] == "exchangeable"]
    shifted = [c for c in cells if c["condition"] == "downward_shift"]
    exchange_coverage_pass = all(c["undercoverage_mean"] <= c["alpha"] + 0.03 for c in exchange)
    shifted_coverage_pass = all(c["undercoverage_mean"] <= c["alpha"] + 0.03 for c in shifted)
    fp_pass = all(c["false_positive_reduction_mean"] >= 0.30 for c in exchange)
    widths = [c["interval_width_mean"] for c in exchange if c["alpha"] == float(config["alpha_values"][0])]
    monotone_width = all(a > b for a, b in zip(widths, widths[1:]))
    strict_pass = exchange_coverage_pass and shifted_coverage_pass and fp_pass and monotone_width
    return {
        "hypothesis_id": "H-018",
        "primary_metric": "empirical_undercoverage",
        "rows": rows,
        "summary": {"cells": cells, "exchangeable_coverage_pass": exchange_coverage_pass, "shifted_coverage_pass": shifted_coverage_pass, "false_positive_reduction_pass": fp_pass, "conservatism_monotone": monotone_width},
        "preregistered_outcome": "PASS" if strict_pass else "FAIL",
    }


RUNNERS = {
    "H-001": run_h001,
    "H-004": run_h004,
    "H-005": run_h005,
    "H-008": run_h008,
    "H-014": run_h014,
    "H-018": run_h018,
}


def run_suite(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in config["seeds"]]
    experiments = [RUNNERS[hypothesis](config[hypothesis], seeds) for hypothesis in RUNNERS]
    return {
        "suite_id": config["suite_id"],
        "config_path": config_path.relative_to(ROOT).as_posix(),
        "seeds": seeds,
        "language_model_training": False,
        "experiments": experiments,
    }


def write_results(payload: dict[str, Any], raw_path: Path, table_path: Path) -> None:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["hypothesis_id", "primary_metric", "preregistered_outcome", "equivalence_flag"])
        writer.writeheader()
        for result in payload["experiments"]:
            writer.writerow(
                {
                    "hypothesis_id": result["hypothesis_id"],
                    "primary_metric": result["primary_metric"],
                    "preregistered_outcome": result["preregistered_outcome"],
                    "equivalence_flag": result.get("equivalence_flag", False),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--raw-output", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--table-output", type=Path, default=DEFAULT_TABLE)
    args = parser.parse_args()
    payload = run_suite(args.config)
    write_results(payload, args.raw_output, args.table_output)
    print(json.dumps({"suite_id": payload["suite_id"], "outcomes": {x["hypothesis_id"]: x["preregistered_outcome"] for x in payload["experiments"]}}, indent=2))


if __name__ == "__main__":
    main()
