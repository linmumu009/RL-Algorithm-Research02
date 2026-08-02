"""Frozen E0 for H-021: a negative-control verifier bridge.

The experiment is analytic/seeded synthetic only.  It estimates a categorical
bridge from sparse clean audits and tests whether that bridge recovers the
clean policy-gradient moment under a latent exploit class.  The formal runner
writes complete raw rows; importing this module never executes the experiment.
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
DEFAULT_CONFIG = ROOT / "06_experiments" / "configs" / "e0_h021.yaml"
DEFAULT_RAW = ROOT / "07_results" / "raw" / "e0_h021_results.json"
DEFAULT_TABLE = ROOT / "07_results" / "tables" / "e0_h021_summary.csv"


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 1.0 if np.allclose(left, right) else 0.0
    return float(np.dot(left, right) / denominator)


def gradient(score: np.ndarray, reward: np.ndarray) -> np.ndarray:
    return np.mean(score * reward[:, None], axis=0)


def mean_ci95(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    mean = float(array.mean())
    if len(array) < 2:
        return {"mean": mean, "lower": mean, "upper": mean}
    half_width = 1.96 * float(array.std(ddof=1)) / math.sqrt(len(array))
    return {"mean": mean, "lower": mean - half_width, "upper": mean + half_width}


def proxy_transition(relevance: float) -> np.ndarray:
    """Return P(proxy=w | latent=u), with rows u and columns w."""

    if not 0.5 <= relevance <= 1.0:
        raise ValueError("Proxy relevance must be in [0.5, 1.0]")
    return np.asarray([[relevance, 1.0 - relevance], [1.0 - relevance, relevance]], dtype=float)


def solve_bridge(moment_matrix: np.ndarray, target_moment: np.ndarray, ridge: float) -> tuple[np.ndarray, float]:
    """Solve E[h(W)|Z]=target by frozen ridge least squares.

    The returned condition number is measured on the unregularized moment
    matrix and capped only to keep the JSON output standards-compliant.
    """

    matrix = np.asarray(moment_matrix, dtype=float)
    target = np.asarray(target_moment, dtype=float)
    if matrix.shape != (2, 2) or target.shape != (2,):
        raise ValueError("The frozen categorical bridge requires a 2x2 moment system")
    raw_condition = float(np.linalg.cond(matrix))
    condition = min(raw_condition, 1.0e12) if np.isfinite(raw_condition) else 1.0e12
    system = matrix.T @ matrix + float(ridge) * np.eye(2)
    solution = np.linalg.solve(system, matrix.T @ target)
    return solution, condition


def _smoothed_mean(values: np.ndarray, prior: float, smoothing: float) -> float:
    return float((values.sum() + smoothing * prior) / (len(values) + smoothing))


def fit_negative_control_bridge(
    clean: np.ndarray,
    observed: np.ndarray,
    diagnostic: np.ndarray,
    verifier_view: np.ndarray,
    context: np.ndarray,
    audited: np.ndarray,
    ridge: float,
    smoothing: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Fit h(W, R, X) from E[Y-h(W,R,X)|Z,R,X]=0 on audits."""

    reward = np.zeros(len(clean), dtype=float)
    conditions: list[float] = []
    audit_prior = float(clean[audited].mean())
    for x_value in (0, 1):
        for r_value in (0, 1):
            stratum = audited & (context == x_value) & (observed == r_value)
            matrix = np.zeros((2, 2), dtype=float)
            target = np.zeros(2, dtype=float)
            for z_value in (0, 1):
                row = stratum & (verifier_view == z_value)
                count = int(row.sum())
                if count == 0:
                    matrix[z_value] = 0.5
                    target[z_value] = audit_prior
                    continue
                target[z_value] = _smoothed_mean(clean[row], audit_prior, smoothing)
                diagnostic_one = _smoothed_mean(diagnostic[row], 0.5, smoothing)
                matrix[z_value] = [1.0 - diagnostic_one, diagnostic_one]
            bridge_values, condition = solve_bridge(matrix, target, ridge)
            conditions.append(condition)
            apply = (context == x_value) & (observed == r_value)
            reward[apply] = bridge_values[diagnostic[apply].astype(int)]
    return reward, {
        "condition_number_mean": float(np.mean(conditions)),
        "condition_number_max": float(np.max(conditions)),
    }


def _table_regression(
    clean: np.ndarray,
    observed: np.ndarray,
    diagnostic: np.ndarray,
    verifier_view: np.ndarray,
    context: np.ndarray,
    audited: np.ndarray,
    smoothing: float,
) -> np.ndarray:
    reward = np.zeros(len(clean), dtype=float)
    prior = float(clean[audited].mean())
    for x_value in (0, 1):
        for r_value in (0, 1):
            for w_value in (0, 1):
                for z_value in (0, 1):
                    cell = (
                        audited
                        & (context == x_value)
                        & (observed == r_value)
                        & (diagnostic == w_value)
                        & (verifier_view == z_value)
                    )
                    estimate = _smoothed_mean(clean[cell], prior, smoothing)
                    apply = (
                        (context == x_value)
                        & (observed == r_value)
                        & (diagnostic == w_value)
                        & (verifier_view == z_value)
                    )
                    reward[apply] = estimate
    return reward


def _observed_stratum_correction(
    clean: np.ndarray,
    observed: np.ndarray,
    context: np.ndarray,
    audited: np.ndarray,
) -> np.ndarray:
    corrected = np.zeros(len(clean), dtype=float)
    for x_value in (0, 1):
        audit_cell = audited & (context == x_value)
        negatives = audit_cell & (clean == 0.0)
        positives = audit_cell & (clean == 1.0)
        false_positive = float(observed[negatives].mean())
        false_negative = float((1.0 - observed[positives]).mean())
        denominator = max(1.0 - false_positive - false_negative, 1.0e-6)
        apply = context == x_value
        corrected[apply] = (observed[apply] - false_positive) / denominator
    return corrected


def _declared_nuisance_projection(observed: np.ndarray, diagnostic: np.ndarray, verifier_view: np.ndarray) -> np.ndarray:
    design = np.column_stack((diagnostic - diagnostic.mean(), verifier_view - verifier_view.mean()))
    coefficients = np.linalg.lstsq(design, observed - observed.mean(), rcond=None)[0]
    return observed - design @ coefficients


def _oracle_reward(
    observed: np.ndarray,
    context: np.ndarray,
    latent: np.ndarray,
    exploit_strength: float,
    base_false_positive: float,
    base_false_negative: float,
) -> np.ndarray:
    clean_probability = sigmoid(-0.35 + 1.05 * context + 0.75 * latent)
    false_positive = base_false_positive + exploit_strength * latent
    false_negative = base_false_negative + exploit_strength * (1 - latent)
    observed_one = clean_probability * (1.0 - false_negative) + (1.0 - clean_probability) * false_positive
    posterior_one = clean_probability * (1.0 - false_negative) / np.maximum(observed_one, 1.0e-12)
    posterior_zero = clean_probability * false_negative / np.maximum(1.0 - observed_one, 1.0e-12)
    return np.where(observed == 1.0, posterior_one, posterior_zero)


def _generate_cell(
    seed: int,
    relevance: float,
    exploit_strength: float,
    config: dict[str, Any],
    invalid_exclusion: bool,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed + int(1000 * relevance) + int(10000 * exploit_strength) + 100000 * invalid_exclusion)
    n = int(config["sample_size"])
    context = rng.integers(0, 2, n)
    latent = rng.integers(0, 2, n)
    context_sign = 2.0 * context - 1.0
    latent_sign = 2.0 * latent - 1.0
    score = np.column_stack((context_sign, latent_sign, context_sign * latent_sign))
    score -= score.mean(axis=0)

    clean_probability = sigmoid(-0.35 + 1.05 * context + 0.75 * latent)
    clean = rng.binomial(1, clean_probability).astype(float)
    false_positive = float(config["base_false_positive"]) + exploit_strength * latent
    false_negative = float(config["base_false_negative"]) + exploit_strength * (1 - latent)
    observed = clean.copy()
    observed[(clean == 0.0) & (rng.random(n) < false_positive)] = 1.0
    observed[(clean == 1.0) & (rng.random(n) < false_negative)] = 0.0

    transition = proxy_transition(relevance)
    diagnostic = rng.binomial(1, transition[latent, 1]).astype(int)
    verifier_view = rng.binomial(1, transition[latent, 1]).astype(int)
    if invalid_exclusion:
        force = (clean == 1.0) & (rng.random(n) < float(config["invalid_exclusion_strength"]))
        diagnostic[force] = 1

    second_fp = float(config["base_false_positive"]) + 0.45 * exploit_strength * latent
    second_fn = float(config["base_false_negative"]) + 0.65 * exploit_strength * (1 - latent)
    paired = clean.copy()
    paired[(clean == 0.0) & (rng.random(n) < second_fp)] = 1.0
    paired[(clean == 1.0) & (rng.random(n) < second_fn)] = 0.0

    audited = np.zeros(n, dtype=bool)
    audited[rng.choice(n, size=int(config["audit_size"]), replace=False)] = True
    bridge, condition = fit_negative_control_bridge(
        clean,
        observed,
        diagnostic,
        verifier_view,
        context,
        audited,
        float(config["bridge_ridge"]),
        float(config["table_smoothing"]),
    )
    baselines = {
        "H001_observed_stratum_correction": _observed_stratum_correction(clean, observed, context, audited),
        "H005_declared_nuisance_projection": _declared_nuisance_projection(observed, diagnostic, verifier_view),
        "direct_proxy_regression": _table_regression(
            clean, observed, diagnostic, verifier_view, context, audited, float(config["table_smoothing"])
        ),
        "direct_pair_average": 0.5 * (observed + paired),
    }
    oracle = _oracle_reward(
        observed,
        context,
        latent,
        exploit_strength,
        float(config["base_false_positive"]),
        float(config["base_false_negative"]),
    )
    clean_gradient = gradient(score, clean)
    bridge_gradient = gradient(score, bridge)
    oracle_gradient = gradient(score, oracle)
    baseline_cosines = {name: cosine(gradient(score, reward), clean_gradient) for name, reward in baselines.items()}
    best_name = max(baseline_cosines, key=baseline_cosines.get)
    bridge_cosine = cosine(bridge_gradient, clean_gradient)
    bridge_error = bridge_gradient - clean_gradient
    return {
        "seed": seed,
        "proxy_relevance": relevance,
        "latent_exploit_strength": exploit_strength,
        "invalid_exclusion": invalid_exclusion,
        "gradient_bias_norm": float(np.linalg.norm(bridge_error)),
        "clean_gradient_cosine": bridge_cosine,
        "cosine_gain_over_best_nonoracle": bridge_cosine - baseline_cosines[best_name],
        "best_nonoracle": best_name,
        "best_nonoracle_cosine": baseline_cosines[best_name],
        "baseline_cosines": baseline_cosines,
        "oracle_latent_exploit_cosine": cosine(oracle_gradient, clean_gradient),
        "bridge_oracle_cosine_difference": abs(bridge_cosine - cosine(oracle_gradient, clean_gradient)),
        "bridge_condition_number": condition["condition_number_max"],
        "bridge_condition_number_mean": condition["condition_number_mean"],
        "false_positive_update_rate": float(np.mean(bridge[clean == 0.0] > 0.5)),
        "bridge_reward_min": float(bridge.min()),
        "bridge_reward_max": float(bridge.max()),
    }


def run_experiment(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in config["seeds"]]
    rows: list[dict[str, Any]] = []
    for relevance in config["proxy_relevance_grid"]:
        for exploit_strength in config["latent_exploit_strength_grid"]:
            for seed in seeds:
                rows.append(_generate_cell(seed, float(relevance), float(exploit_strength), config, False))
    for seed in seeds:
        rows.append(_generate_cell(seed, 0.80, 0.30, config, True))

    cells: list[dict[str, Any]] = []
    valid_rows = [row for row in rows if not row["invalid_exclusion"]]
    for relevance in config["proxy_relevance_grid"]:
        for exploit_strength in config["latent_exploit_strength_grid"]:
            subset = [
                row
                for row in valid_rows
                if row["proxy_relevance"] == float(relevance)
                and row["latent_exploit_strength"] == float(exploit_strength)
            ]
            cells.append(
                {
                    "proxy_relevance": float(relevance),
                    "latent_exploit_strength": float(exploit_strength),
                    "gradient_bias_norm_mean": float(np.mean([row["gradient_bias_norm"] for row in subset])),
                    "clean_gradient_cosine_mean": float(np.mean([row["clean_gradient_cosine"] for row in subset])),
                    "cosine_gain_ci95": mean_ci95(row["cosine_gain_over_best_nonoracle"] for row in subset),
                    "bridge_condition_number_mean": float(np.mean([row["bridge_condition_number"] for row in subset])),
                    "bridge_oracle_cosine_difference_mean": float(
                        np.mean([row["bridge_oracle_cosine_difference"] for row in subset])
                    ),
                }
            )

    thresholds = config["success_thresholds"]
    strong = [cell for cell in cells if cell["proxy_relevance"] >= float(config["strong_proxy_threshold"])]
    strong_latent = [cell for cell in strong if cell["latent_exploit_strength"] > 0.0]
    weak = [cell for cell in cells if cell["proxy_relevance"] < float(config["strong_proxy_threshold"])]
    strong_bias_pass = all(cell["gradient_bias_norm_mean"] < float(thresholds["strong_proxy_bias_max"]) for cell in strong)
    gain_cells = sum(
        cell["cosine_gain_ci95"]["mean"] >= float(thresholds["cosine_gain_min"])
        and cell["cosine_gain_ci95"]["lower"] > 0.0
        for cell in strong_latent
    )
    oracle_control_pass = all(
        cell["bridge_oracle_cosine_difference_mean"] <= float(thresholds["revealed_oracle_cosine_difference_max"])
        for cell in strong_latent
    )
    strong_condition = float(np.median([cell["bridge_condition_number_mean"] for cell in strong]))
    weak_condition = float(np.median([cell["bridge_condition_number_mean"] for cell in weak]))
    condition_degradation_pass = weak_condition >= float(thresholds["weak_condition_ratio_min"]) * strong_condition
    invalid_rows = [row for row in rows if row["invalid_exclusion"]]
    invalid_bias = float(np.mean([row["gradient_bias_norm"] for row in invalid_rows]))
    matched_valid_bias = float(
        np.mean(
            [
                row["gradient_bias_norm"]
                for row in valid_rows
                if row["proxy_relevance"] == 0.80 and row["latent_exploit_strength"] == 0.30
            ]
        )
    )
    passed = (
        strong_bias_pass
        and gain_cells >= int(thresholds["minimum_gain_cells"])
        and oracle_control_pass
        and condition_degradation_pass
    )
    return {
        "experiment_id": config["experiment_id"],
        "hypothesis_id": "H-021",
        "config_path": config_path.relative_to(ROOT).as_posix(),
        "seeds": seeds,
        "language_model_training": False,
        "rows": rows,
        "summary": {
            "cells": cells,
            "strong_proxy_bias_pass": strong_bias_pass,
            "strong_latent_gain_cells": gain_cells,
            "revealed_latent_oracle_control_pass": oracle_control_pass,
            "condition_degradation_pass": condition_degradation_pass,
            "strong_condition_median": strong_condition,
            "weak_condition_median": weak_condition,
            "invalid_exclusion_bias_mean": invalid_bias,
            "matched_valid_bias_mean": matched_valid_bias,
        },
        "preregistered_outcome": "PASS" if passed else "FAIL",
    }


def write_results(payload: dict[str, Any], raw_path: Path, table_path: Path) -> None:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "proxy_relevance",
            "latent_exploit_strength",
            "gradient_bias_norm_mean",
            "clean_gradient_cosine_mean",
            "cosine_gain_mean",
            "cosine_gain_ci95_lower",
            "bridge_condition_number_mean",
            "bridge_oracle_cosine_difference_mean",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for cell in payload["summary"]["cells"]:
            writer.writerow(
                {
                    "proxy_relevance": cell["proxy_relevance"],
                    "latent_exploit_strength": cell["latent_exploit_strength"],
                    "gradient_bias_norm_mean": cell["gradient_bias_norm_mean"],
                    "clean_gradient_cosine_mean": cell["clean_gradient_cosine_mean"],
                    "cosine_gain_mean": cell["cosine_gain_ci95"]["mean"],
                    "cosine_gain_ci95_lower": cell["cosine_gain_ci95"]["lower"],
                    "bridge_condition_number_mean": cell["bridge_condition_number_mean"],
                    "bridge_oracle_cosine_difference_mean": cell["bridge_oracle_cosine_difference_mean"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--raw-output", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--table-output", type=Path, default=DEFAULT_TABLE)
    args = parser.parse_args()
    payload = run_experiment(args.config)
    write_results(payload, args.raw_output, args.table_output)
    print(json.dumps({"experiment_id": payload["experiment_id"], "outcome": payload["preregistered_outcome"]}, indent=2))


if __name__ == "__main__":
    main()
