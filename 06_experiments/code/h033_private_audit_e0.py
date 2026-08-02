"""Frozen E0 for H-033: privacy-stable reusable audit gradients.

The formal experiment is seeded synthetic and never trains a language model.
It exposes a fixed binary audit table only through clipped vector queries,
calibrates Gaussian releases with zCDP composition, and stops at the frozen
privacy budget. Importing this module never executes or writes the experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "06_experiments" / "configs" / "e0_h033.yaml"
DEFAULT_RAW = ROOT / "07_results" / "raw" / "e0_h033_results.json"
DEFAULT_TABLE = ROOT / "07_results" / "tables" / "e0_h033_summary.csv"
H033_METHOD = "privacy_stable_reusable_audit_gradient"
EXACT_NONORACLE_COMPARATORS = (
    "reusable_holdout_thresholdout",
    "generic_holdout_binary_release",
    "once_trained_private_reward_model",
)


def unit(vector: np.ndarray, tolerance: float = 1.0e-12) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    return np.zeros_like(vector) if norm <= tolerance else vector / norm


def cosine(left: np.ndarray, right: np.ndarray, tolerance: float = 1.0e-12) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= tolerance:
        return 0.0
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def clip_rows(rows: np.ndarray, l2_clip: float) -> np.ndarray:
    rows = np.asarray(rows, dtype=float)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    scales = np.minimum(1.0, float(l2_clip) / np.maximum(norms, 1.0e-300))
    return rows * scales


def gaussian_sigma(l2_sensitivity: float, rho: float) -> float:
    """Gaussian standard deviation for a rho-zCDP vector release."""

    if rho <= 0.0:
        raise ValueError("rho must be positive")
    return float(l2_sensitivity) / math.sqrt(2.0 * float(rho))


@dataclass
class RhoAccountant:
    budget: float
    spent: float = 0.0
    tolerance: float = 1.0e-12

    def can_release(self, rho_cost: float) -> bool:
        return self.spent + float(rho_cost) <= self.budget + self.tolerance

    def spend(self, rho_cost: float) -> None:
        if not self.can_release(rho_cost):
            raise RuntimeError("privacy filter exhausted")
        self.spent += float(rho_cost)


@dataclass(frozen=True)
class BinaryAuditTable:
    features: np.ndarray
    labels: np.ndarray
    signed_operator: np.ndarray

    @property
    def size(self) -> int:
        return int(len(self.labels))


def _rng(seed: int, *coordinates: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), *map(int, coordinates)]))


def make_binary_table(seed: int, size: int, dimension: int, salt: int) -> BinaryAuditTable:
    rng = _rng(seed, size, dimension, salt)
    features = rng.choice((-1.0, 1.0), size=(int(size), int(dimension))) / math.sqrt(float(dimension))
    labels = rng.choice((-1.0, 1.0), size=int(size))
    operator = features.T @ (labels[:, None] * features) / float(size)
    return BinaryAuditTable(features=features, labels=labels, signed_operator=operator)


def replace_labels(table: BinaryAuditTable, labels: np.ndarray) -> BinaryAuditTable:
    labels = np.asarray(labels, dtype=float)
    operator = table.features.T @ (labels[:, None] * table.features) / float(table.size)
    return BinaryAuditTable(features=table.features, labels=labels, signed_operator=operator)


def signal_vector(dimension: int, magnitude: float) -> np.ndarray:
    return float(magnitude) * unit(np.linspace(1.0, 2.0, int(dimension)))


def candidate_pool(seed: int, dimension: int, count: int) -> np.ndarray:
    rng = _rng(seed, dimension, count, 7001)
    candidates = rng.normal(size=(int(count), int(dimension)))
    return np.asarray([unit(candidate) for candidate in candidates])


def query_contributions(
    table: BinaryAuditTable,
    theta: np.ndarray,
    signal: np.ndarray,
    operator_scale: float,
) -> np.ndarray:
    projections = table.features @ np.asarray(theta, dtype=float)
    return signal + float(operator_scale) * table.labels[:, None] * table.features * projections[:, None]


def clipped_query_mean(
    table: BinaryAuditTable,
    theta: np.ndarray,
    signal: np.ndarray,
    operator_scale: float,
    l2_clip: float,
) -> np.ndarray:
    # Every normal-grid feature has unit norm and |x^T theta| <= 1. The
    # shortcut is therefore exact whenever the declared triangle bound fits.
    if float(np.linalg.norm(signal)) + abs(float(operator_scale)) <= float(l2_clip):
        return signal + float(operator_scale) * table.signed_operator @ np.asarray(theta, dtype=float)
    return np.mean(
        clip_rows(query_contributions(table, theta, signal, operator_scale), l2_clip),
        axis=0,
    )


def split_tables(table: BinaryAuditTable, minimum_partition_size: int) -> list[BinaryAuditTable]:
    partitions: list[BinaryAuditTable] = []
    for start in range(0, table.size, int(minimum_partition_size)):
        end = min(start + int(minimum_partition_size), table.size)
        if end - start < int(minimum_partition_size):
            break
        features = table.features[start:end]
        labels = table.labels[start:end]
        operator = features.T @ (labels[:, None] * features) / float(end - start)
        partitions.append(BinaryAuditTable(features, labels, operator))
    return partitions


def _table_slice(table: BinaryAuditTable, start: int, end: int) -> BinaryAuditTable:
    features = table.features[start:end]
    labels = table.labels[start:end]
    operator = features.T @ (labels[:, None] * features) / float(end - start)
    return BinaryAuditTable(features, labels, operator)


def _private_lookup(
    audit: BinaryAuditTable,
    candidates: np.ndarray,
    signal: np.ndarray,
    operator_scale: float,
    l2_clip: float,
    rho_budget: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    exact = np.asarray(
        [clipped_query_mean(audit, theta, signal, operator_scale, l2_clip) for theta in candidates]
    )
    sensitivity = 2.0 * float(l2_clip) * math.sqrt(float(len(candidates))) / float(audit.size)
    sigma = gaussian_sigma(sensitivity, rho_budget)
    return exact + rng.normal(0.0, sigma, size=exact.shape), sigma


def _digest(indices: list[int]) -> str:
    payload = ",".join(map(str, indices)).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def run_trace(
    method: str,
    audit: BinaryAuditTable,
    population: BinaryAuditTable,
    candidates: np.ndarray,
    query_count: int,
    rho_budget: float,
    signal: np.ndarray,
    config: dict[str, Any],
    seed: int,
    *,
    adaptive: bool = True,
    privacy_off: bool = False,
    operator_scale: float | None = None,
) -> dict[str, Any]:
    dimension = int(candidates.shape[1])
    scale = float(config["operator_scale"] if operator_scale is None else operator_scale)
    l2_clip = float(config["contribution_l2_clip"])
    tolerance = float(config["privacy_tolerance"])
    rng = _rng(seed, audit.size, dimension, int(query_count), int(round(1000.0 * rho_budget)), sum(map(ord, method)))
    accountant = RhoAccountant(float(rho_budget), tolerance=tolerance)
    rho_per_query = float(rho_budget) / float(query_count)
    sensitivity = 2.0 * l2_clip / float(audit.size)
    sigma = 0.0 if privacy_off else gaussian_sigma(sensitivity, rho_per_query)
    state = np.zeros(dimension, dtype=float)
    indices: list[int] = []
    biases: list[float] = []
    cosines: list[float] = []
    false_positives: list[float] = []
    generalization_gaps: list[float] = []
    thresholdout_overrides = 0
    generic_accepts = 0

    partitions = split_tables(audit, int(config["disjoint_min_partition_size"]))
    split = int(round(audit.size * (1.0 - float(config["thresholdout_holdout_fraction"]))))
    split = min(max(split, 1), audit.size - 1)
    train = _table_slice(audit, 0, split)
    holdout = _table_slice(audit, split, audit.size)
    fixed_h001 = clipped_query_mean(audit, candidates[0], signal, scale, l2_clip)
    private_lookup = None
    private_sigma = 0.0
    if method == "once_trained_private_reward_model":
        private_lookup, private_sigma = _private_lookup(
            audit, candidates, signal, scale, l2_clip, rho_budget, rng
        )

    for round_index in range(int(query_count)):
        if adaptive and float(np.linalg.norm(state)) > tolerance:
            candidate_index = int(np.argmax(candidates @ state))
        else:
            candidate_index = round_index % len(candidates)
        theta = candidates[candidate_index]
        population_truth = clipped_query_mean(population, theta, signal, scale, l2_clip)
        audit_exact = clipped_query_mean(audit, theta, signal, scale, l2_clip)
        release: np.ndarray | None

        if method == "naive_exact_audit_reuse":
            release = audit_exact
        elif method == "fresh_audit_oracle":
            release = population_truth
        elif method == "disjoint_sample_splitting":
            release = (
                clipped_query_mean(partitions[round_index], theta, signal, scale, l2_clip)
                if round_index < len(partitions)
                else None
            )
        elif method == "reusable_holdout_thresholdout":
            train_answer = clipped_query_mean(train, theta, signal, scale, l2_clip)
            holdout_answer = clipped_query_mean(holdout, theta, signal, scale, l2_clip)
            discrepancy = float(np.linalg.norm(train_answer - holdout_answer))
            if (
                discrepancy > float(config["thresholdout_discrepancy_threshold"])
                and thresholdout_overrides < int(config["thresholdout_max_overrides"])
            ):
                release = holdout_answer + rng.normal(
                    0.0, float(config["thresholdout_noise_std"]), size=dimension
                )
                thresholdout_overrides += 1
            else:
                release = train_answer
        elif method == "generic_holdout_binary_release":
            train_answer = clipped_query_mean(train, theta, signal, scale, l2_clip)
            holdout_answer = clipped_query_mean(holdout, theta, signal, scale, l2_clip)
            accepted = float(np.linalg.norm(train_answer - holdout_answer)) <= float(
                config["generic_holdout_discrepancy_threshold"]
            )
            generic_accepts += int(accepted)
            release = train_answer if accepted else np.zeros(dimension, dtype=float)
        elif method == "once_trained_private_reward_model":
            assert private_lookup is not None
            release = private_lookup[candidate_index]
        elif method == "H001_single_query_correction":
            release = fixed_h001
        elif method == H033_METHOD:
            if not privacy_off and not accountant.can_release(rho_per_query):
                release = None
            else:
                release = audit_exact.copy()
                if not privacy_off:
                    release += rng.normal(0.0, sigma, size=dimension)
                    accountant.spend(rho_per_query)
        else:
            raise ValueError(f"Unknown method: {method}")

        if release is None:
            continue
        indices.append(candidate_index)
        bias = float(np.linalg.norm(release - population_truth))
        biases.append(bias)
        cosines.append(cosine(release, population_truth, tolerance))
        false_positives.append(float(np.dot(release, population_truth) <= 0.0))
        generalization_gaps.append(float(np.linalg.norm(audit_exact - population_truth)))
        if adaptive:
            state += float(config["adaptation_rate"]) * (release - signal)

    return {
        "method": method,
        "adaptive": bool(adaptive),
        "requested_rounds": int(query_count),
        "usable_release_rounds": len(indices),
        "population_gradient_bias_mean": _mean(biases),
        "population_gradient_bias_max": max(biases, default=0.0),
        "clean_gradient_cosine_mean": _mean(cosines),
        "clean_gradient_cosine_min": min(cosines, default=0.0),
        "false_positive_direction_rate": _mean(false_positives),
        "audit_generalization_gap_mean": _mean(generalization_gaps),
        "privacy_rho_spent": float(accountant.spent),
        "gaussian_sigma": float(private_sigma if method == "once_trained_private_reward_model" else sigma),
        "thresholdout_overrides": thresholdout_overrides,
        "generic_holdout_accepts": generic_accepts,
        "unique_query_candidates": len(set(indices)),
        "candidate_sequence_sha256": _digest(indices),
    }


def _row_coordinates(
    seed: int, audit_size: int, query_count: int, rho_budget: float, dimension: int
) -> dict[str, Any]:
    return {
        "seed": seed,
        "audit_size": audit_size,
        "adaptive_query_count": query_count,
        "zcdp_rho_budget": rho_budget,
        "score_dimension": dimension,
    }


def _annotate_gains(traces: list[dict[str, Any]]) -> None:
    by_method = {trace["method"]: trace for trace in traces}
    target = by_method[H033_METHOD]
    naive = by_method["naive_exact_audit_reuse"]
    best_name = max(
        EXACT_NONORACLE_COMPARATORS,
        key=lambda name: (
            by_method[name]["clean_gradient_cosine_mean"],
            -by_method[name]["false_positive_direction_rate"],
        ),
    )
    best = by_method[best_name]
    target.update(
        {
            "best_exact_nonoracle": best_name,
            "cosine_gain_over_naive": target["clean_gradient_cosine_mean"]
            - naive["clean_gradient_cosine_mean"],
            "false_positive_rate_reduction_over_naive": naive["false_positive_direction_rate"]
            - target["false_positive_direction_rate"],
            "cosine_gain_over_best_exact_nonoracle": target["clean_gradient_cosine_mean"]
            - best["clean_gradient_cosine_mean"],
            "false_positive_rate_reduction_over_best_exact_nonoracle": best[
                "false_positive_direction_rate"
            ]
            - target["false_positive_direction_rate"],
            "usable_round_ratio_over_disjoint": target["usable_release_rounds"]
            / max(by_method["disjoint_sample_splitting"]["usable_release_rounds"], 1),
        }
    )


def _control_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = int(config["control_audit_size"])
    q = int(config["control_query_count"])
    rho = float(config["control_rho_budget"])
    d = int(config["control_score_dimension"])
    for seed in map(int, config["seeds"]):
        audit = make_binary_table(seed, n, d, 33001)
        population = make_binary_table(seed, int(config["population_size"]), d, 33002)
        candidates = candidate_pool(seed, d, int(config["query_candidate_count"]))
        signal = signal_vector(d, float(config["signal_magnitude"]))

        nonadaptive = run_trace(
            H033_METHOD, audit, population, candidates, q, rho, signal, config, seed, adaptive=False
        )
        rows.append({"control": "nonadaptive_query_sequence", "seed": seed, **nonadaptive})

        privacy_off = run_trace(
            H033_METHOD,
            audit,
            population,
            candidates,
            q,
            rho,
            signal,
            config,
            seed,
            privacy_off=True,
        )
        naive = run_trace(
            "naive_exact_audit_reuse", audit, population, candidates, q, rho, signal, config, seed
        )
        rows.append(
            {
                "control": "privacy_off_limit",
                "seed": seed,
                **privacy_off,
                "naive_bias_difference": privacy_off["population_gradient_bias_mean"]
                - naive["population_gradient_bias_mean"],
                "naive_cosine_difference": privacy_off["clean_gradient_cosine_mean"]
                - naive["clean_gradient_cosine_mean"],
                "candidate_sequence_matches_naive": privacy_off["candidate_sequence_sha256"]
                == naive["candidate_sequence_sha256"],
            }
        )

        zero_signal = np.zeros(d, dtype=float)
        zero_population = replace_labels(population, np.zeros(population.size, dtype=float))
        zero = run_trace(
            H033_METHOD, audit, zero_population, candidates, q, rho, zero_signal, config, seed
        )
        rows.append({"control": "zero_population_gradient", "seed": seed, **zero})

        permuted_rng = _rng(seed, 33003)
        permuted = replace_labels(audit, permuted_rng.permutation(audit.labels))
        permuted_trace = run_trace(
            H033_METHOD, permuted, population, candidates, q, rho, signal, config, seed
        )
        rows.append({"control": "permuted_audit_labels", "seed": seed, **permuted_trace})

        violation_scale = float(config["clip_violation_operator_scale"])
        violation = run_trace(
            H033_METHOD,
            audit,
            population,
            candidates,
            q,
            rho,
            signal,
            config,
            seed,
            operator_scale=violation_scale,
        )
        sample_rows = query_contributions(audit, candidates[0], signal, violation_scale)
        rows.append(
            {
                "control": "contribution_clip_violation",
                "seed": seed,
                **violation,
                "unclipped_contribution_norm_max": float(np.linalg.norm(sample_rows, axis=1).max()),
                "clipped_contribution_norm_max": float(
                    np.linalg.norm(clip_rows(sample_rows, float(config["contribution_l2_clip"])), axis=1).max()
                ),
            }
        )

        drift_labels = np.ones(population.size, dtype=float)
        drift_population = replace_labels(population, drift_labels)
        drift = run_trace(
            H033_METHOD, audit, drift_population, candidates, q, rho, signal, config, seed
        )
        rows.append({"control": "population_drift_after_audit", "seed": seed, **drift})

        equal = run_trace(
            H033_METHOD, audit, population, candidates, q, rho, signal, config, seed
        )
        rows.append(
            {
                "control": "equal_audit_rows_and_query_candidates",
                "seed": seed,
                **equal,
                "audit_rows": audit.size,
                "query_candidates": len(candidates),
            }
        )
    return rows


def _aggregate_cells(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    keys = (
        "audit_size",
        "adaptive_query_count",
        "zcdp_rho_budget",
        "score_dimension",
        "method",
    )
    cells: list[dict[str, Any]] = []
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    thresholds = config["success_thresholds"]
    for coordinates, subset in sorted(groups.items()):
        cell = dict(zip(keys, coordinates))
        cell.update(
            {
                "population_gradient_bias_mean": _mean(
                    row["population_gradient_bias_mean"] for row in subset
                ),
                "clean_gradient_cosine_mean": _mean(row["clean_gradient_cosine_mean"] for row in subset),
                "false_positive_direction_rate": _mean(
                    row["false_positive_direction_rate"] for row in subset
                ),
                "usable_release_rounds_mean": _mean(row["usable_release_rounds"] for row in subset),
                "privacy_rho_spent_mean": _mean(row["privacy_rho_spent"] for row in subset),
            }
        )
        if cell["method"] == H033_METHOD:
            for metric in (
                "cosine_gain_over_naive",
                "false_positive_rate_reduction_over_naive",
                "cosine_gain_over_best_exact_nonoracle",
                "false_positive_rate_reduction_over_best_exact_nonoracle",
                "usable_round_ratio_over_disjoint",
            ):
                cell[f"{metric}_mean"] = _mean(row[metric] for row in subset)
            beats_naive = (
                cell["cosine_gain_over_naive_mean"] >= float(thresholds["cosine_gain_min"])
                or cell["false_positive_rate_reduction_over_naive_mean"]
                >= float(thresholds["false_positive_rate_reduction_min"])
            )
            beats_best = (
                cell["cosine_gain_over_best_exact_nonoracle_mean"]
                >= float(thresholds["cosine_gain_min"])
                or cell["false_positive_rate_reduction_over_best_exact_nonoracle_mean"]
                >= float(thresholds["false_positive_rate_reduction_min"])
            )
            cell["qualifying_adaptive_gain"] = bool(
                cell["population_gradient_bias_mean"]
                <= float(thresholds["population_gradient_bias_max"])
                and cell["false_positive_direction_rate"]
                <= float(thresholds["false_positive_direction_rate_max"])
                and cell["clean_gradient_cosine_mean"]
                >= float(thresholds["clean_gradient_cosine_min"])
                and cell["usable_round_ratio_over_disjoint_mean"]
                >= float(thresholds["usable_round_ratio_over_disjoint_min"])
                and beats_naive
                and beats_best
            )
        cells.append(cell)
    return cells


def run_experiment(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    methods = [*config["baselines"], H033_METHOD]
    rows: list[dict[str, Any]] = []
    for audit_size in map(int, config["audit_size_grid"]):
        for query_count in map(int, config["adaptive_query_count_grid"]):
            for rho_budget in map(float, config["zcdp_rho_budget_grid"]):
                for dimension in map(int, config["score_dimension_grid"]):
                    for seed in map(int, config["seeds"]):
                        audit = make_binary_table(seed, audit_size, dimension, 33011)
                        population = make_binary_table(
                            seed, int(config["population_size"]), dimension, 33012
                        )
                        candidates = candidate_pool(seed, dimension, int(config["query_candidate_count"]))
                        signal = signal_vector(dimension, float(config["signal_magnitude"]))
                        traces = [
                            run_trace(
                                method,
                                audit,
                                population,
                                candidates,
                                query_count,
                                rho_budget,
                                signal,
                                config,
                                seed,
                            )
                            for method in methods
                        ]
                        _annotate_gains(traces)
                        coordinates = _row_coordinates(
                            seed, audit_size, query_count, rho_budget, dimension
                        )
                        rows.extend([{**coordinates, **trace} for trace in traces])

    cells = _aggregate_cells(rows, config)
    h033_cells = [cell for cell in cells if cell["method"] == H033_METHOD]
    thresholds = config["success_thresholds"]
    qualifying = [cell for cell in h033_cells if cell["qualifying_adaptive_gain"]]
    safety_pass = all(
        cell["population_gradient_bias_mean"] <= float(thresholds["population_gradient_bias_max"])
        and cell["false_positive_direction_rate"]
        <= float(thresholds["false_positive_direction_rate_max"])
        and cell["clean_gradient_cosine_mean"] >= float(thresholds["clean_gradient_cosine_min"])
        for cell in h033_cells
    )
    round_efficiency_pass = all(
        cell["usable_round_ratio_over_disjoint_mean"]
        >= float(thresholds["usable_round_ratio_over_disjoint_min"])
        for cell in h033_cells
    )
    passed = (
        safety_pass
        and round_efficiency_pass
        and len(qualifying) >= int(thresholds["minimum_adaptive_gain_cells"])
    )
    controls = _control_rows(config)
    return {
        "experiment_id": config["experiment_id"],
        "hypothesis_id": "H-033",
        "config_path": config_path.relative_to(ROOT).as_posix(),
        "seeds": [int(seed) for seed in config["seeds"]],
        "language_model_training": False,
        "metric_aggregation": "per-seed trace means followed by equal-weight seed means",
        "rows": rows,
        "control_rows": controls,
        "summary": {
            "cells": cells,
            "h033_adaptive_cells": len(h033_cells),
            "qualifying_adaptive_gain_cells": len(qualifying),
            "safety_pass": safety_pass,
            "round_efficiency_pass": round_efficiency_pass,
            "nonadaptive_controls_excluded_from_gain_claim": True,
        },
        "preregistered_outcome": "PASS" if passed else "FAIL",
    }


def write_results(payload: dict[str, Any], raw_path: Path, table_path: Path) -> None:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    fields = [
        "audit_size",
        "adaptive_query_count",
        "zcdp_rho_budget",
        "score_dimension",
        "method",
        "population_gradient_bias_mean",
        "clean_gradient_cosine_mean",
        "false_positive_direction_rate",
        "usable_release_rounds_mean",
        "privacy_rho_spent_mean",
    ]
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for cell in payload["summary"]["cells"]:
            writer.writerow({field: cell[field] for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--raw-output", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--table-output", type=Path, default=DEFAULT_TABLE)
    args = parser.parse_args()
    payload = run_experiment(args.config)
    write_results(payload, args.raw_output, args.table_output)
    print(
        json.dumps(
            {"experiment_id": payload["experiment_id"], "outcome": payload["preregistered_outcome"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
