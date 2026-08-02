"""Frozen E0 for H-039: channel-set advantage sign certificates.

The experiment is analytic/seeded synthetic and never trains a language
model.  Contextual FP/FN confidence rectangles are enumerated jointly by
context.  Every corner is mapped through the same group-centering operation,
so a completion interval never combines mutually incompatible group
baselines.  Importing this module does not execute or write formal results.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "06_experiments" / "configs" / "e0_h039.yaml"
DEFAULT_RAW = ROOT / "07_results" / "raw" / "e0_h039_results.json"
DEFAULT_TABLE = ROOT / "07_results" / "tables" / "e0_h039_summary.csv"
H039_METHOD = "channel_set_advantage_sign_certificate"


def unit(vector: np.ndarray, tolerance: float = 1.0e-12) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    return np.zeros_like(vector) if norm <= tolerance else vector / norm


def cosine(left: np.ndarray, right: np.ndarray, tolerance: float = 1.0e-12) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= tolerance:
        return 0.0
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def group_center(values: np.ndarray, group_ids: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    group_ids = np.asarray(group_ids)
    centered = np.empty_like(values)
    for group_id in np.unique(group_ids):
        mask = group_ids == group_id
        centered[mask] = values[mask] - float(np.mean(values[mask]))
    return centered


def inverse_binary_channel(
    observed_reward: np.ndarray,
    context_ids: np.ndarray,
    channels: np.ndarray,
    tolerance: float = 1.0e-12,
) -> np.ndarray:
    """Return the unbiased clean-label surrogate for contextual FP/FN rates."""

    observed_reward = np.asarray(observed_reward, dtype=float)
    context_ids = np.asarray(context_ids, dtype=int)
    channels = np.asarray(channels, dtype=float)
    false_positive = channels[context_ids, 0]
    false_negative = channels[context_ids, 1]
    youden = 1.0 - false_positive - false_negative
    if np.any(youden <= tolerance):
        raise ValueError("Every contextual channel must have positive Youden index")
    return (observed_reward - false_positive) / youden


def enumerate_channel_corners(intervals: np.ndarray) -> np.ndarray:
    """Enumerate four FP/FN corners per context, jointly across contexts."""

    intervals = np.asarray(intervals, dtype=float)
    if intervals.ndim != 3 or intervals.shape[1:] != (2, 2):
        raise ValueError("intervals must have shape [contexts, fp_or_fn, lower_or_upper]")
    if np.any(intervals[:, :, 0] > intervals[:, :, 1]):
        raise ValueError("channel interval lower bound exceeds upper bound")
    configurations: list[np.ndarray] = []
    for choices in itertools.product(range(4), repeat=intervals.shape[0]):
        channels = np.empty((intervals.shape[0], 2), dtype=float)
        for context, choice in enumerate(choices):
            channels[context, 0] = intervals[context, 0, (choice >> 1) & 1]
            channels[context, 1] = intervals[context, 1, choice & 1]
        configurations.append(channels)
    return np.asarray(configurations)


def advantage_configurations(
    observed_reward: np.ndarray,
    context_ids: np.ndarray,
    group_ids: np.ndarray,
    channel_intervals: np.ndarray,
    tolerance: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray]:
    channel_corners = enumerate_channel_corners(channel_intervals)
    advantages = np.asarray(
        [
            group_center(
                inverse_binary_channel(observed_reward, context_ids, channels, tolerance),
                group_ids,
            )
            for channels in channel_corners
        ]
    )
    return advantages, channel_corners


def advantage_intervals(advantage_vertices: np.ndarray) -> np.ndarray:
    advantage_vertices = np.asarray(advantage_vertices, dtype=float)
    return np.column_stack((np.min(advantage_vertices, axis=0), np.max(advantage_vertices, axis=0)))


def certified_weights(intervals: np.ndarray, tolerance: float = 1.0e-12) -> np.ndarray:
    """Use the signed worst-case absolute margin only for an identified sign."""

    intervals = np.asarray(intervals, dtype=float)
    lower = intervals[:, 0]
    upper = intervals[:, 1]
    weights = np.zeros(len(intervals), dtype=float)
    weights[lower > tolerance] = lower[lower > tolerance]
    weights[upper < -tolerance] = upper[upper < -tolerance]
    return weights


def gradient_from_weights(weights: np.ndarray, scores: np.ndarray) -> np.ndarray:
    return np.mean(np.asarray(weights, dtype=float)[:, None] * np.asarray(scores, dtype=float), axis=0)


def _cross(origin: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    a = left - origin
    b = right - origin
    return float(a[0] * b[1] - a[1] * b[0])


def convex_hull(points: Sequence[Sequence[float]], tolerance: float = 1.0e-12) -> np.ndarray:
    unique = sorted({(float(point[0]), float(point[1])) for point in points})
    if len(unique) <= 1:
        return np.asarray(unique, dtype=float)
    arrays = [np.asarray(point) for point in unique]
    lower: list[np.ndarray] = []
    upper: list[np.ndarray] = []
    for point in arrays:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= tolerance:
            lower.pop()
        lower.append(point)
    for point in reversed(arrays):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= tolerance:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def _point_in_hull(point: np.ndarray, hull: np.ndarray, tolerance: float) -> bool:
    if len(hull) == 0:
        return False
    if len(hull) == 1:
        return bool(np.linalg.norm(point - hull[0]) <= tolerance)
    if len(hull) == 2:
        segment = hull[1] - hull[0]
        denominator = float(np.dot(segment, segment))
        coefficient = 0.0 if denominator == 0.0 else float(np.dot(point - hull[0], segment) / denominator)
        projection = hull[0] + np.clip(coefficient, 0.0, 1.0) * segment
        return bool(np.linalg.norm(point - projection) <= tolerance)
    signs = [_cross(hull[index], hull[(index + 1) % len(hull)], point) for index in range(len(hull))]
    return all(value >= -tolerance for value in signs) or all(value <= tolerance for value in signs)


def minimum_norm_direction(points: np.ndarray, tolerance: float = 1.0e-12) -> tuple[np.ndarray, bool]:
    hull = convex_hull(points, tolerance)
    origin = np.zeros(2, dtype=float)
    if _point_in_hull(origin, hull, tolerance):
        return origin, True
    if len(hull) == 1:
        return unit(hull[0], tolerance), False
    candidates: list[np.ndarray] = []
    for index in range(len(hull)):
        start = hull[index]
        segment = hull[(index + 1) % len(hull)] - start
        denominator = float(np.dot(segment, segment))
        coefficient = 0.0 if denominator == 0.0 else float(np.clip(-np.dot(start, segment) / denominator, 0.0, 1.0))
        candidates.append(start + coefficient * segment)
    closest = min(candidates, key=lambda value: float(np.dot(value, value)))
    return unit(closest, tolerance), False


def _channel_truth(context_count: int, heterogeneity: float, config: dict[str, Any]) -> np.ndarray:
    pattern = np.linspace(-1.0, 1.0, context_count)
    shift = 0.5 * float(heterogeneity) * pattern
    fp = np.clip(
        float(config["base_false_positive_rate"]) + shift,
        0.01,
        float(config["maximum_channel_rate"]),
    )
    fn = np.clip(
        float(config["base_false_negative_rate"]) - shift,
        0.01,
        float(config["maximum_channel_rate"]),
    )
    return np.column_stack((fp, fn))


def _audit_intervals(
    truth: np.ndarray,
    half_width: float,
    audit_size: int,
    rng: np.random.Generator,
    config: dict[str, Any],
    *,
    misspecified: bool = False,
) -> np.ndarray:
    scale = float(config["audit_midpoint_offset_fraction"]) * math.sqrt(64.0 / float(audit_size))
    offsets = rng.uniform(-scale, scale, size=truth.shape) * float(half_width)
    if misspecified:
        offsets = np.full_like(truth, 1.25 * float(half_width))
    midpoint = truth + offsets
    lower = np.clip(midpoint - float(half_width), 0.0, float(config["maximum_channel_rate"]))
    upper = np.clip(midpoint + float(half_width), 0.0, float(config["maximum_channel_rate"]))
    return np.stack((lower, upper), axis=2)


def make_problem(
    seed: int,
    context_count: int,
    audit_size: int,
    interval_half_width: float,
    channel_heterogeneity: float,
    score_angle: float,
    config: dict[str, Any],
    *,
    misspecified: bool = False,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [seed, context_count, audit_size, int(interval_half_width * 1000), int(channel_heterogeneity * 1000), int(score_angle)]
        )
    )
    group_count = int(config["group_count"])
    repetitions = int(config["completions_per_context_per_group"])
    pattern = np.linspace(-1.0, 1.0, repetitions)
    context_pattern = np.linspace(-1.0, 1.0, context_count)
    group_ids: list[int] = []
    context_ids: list[int] = []
    clean_probability: list[float] = []
    scores: list[np.ndarray] = []
    for group in range(group_count):
        for context in range(context_count):
            for repetition, signed_position in enumerate(pattern):
                group_ids.append(group)
                context_ids.append(context)
                clean_probability.append(
                    float(config["clean_probability_center"])
                    + float(config["clean_advantage_amplitude"]) * signed_position
                    + float(config["context_clean_shift"]) * context_pattern[context]
                )
                phase = signed_position + 0.35 * context_pattern[context]
                angle = math.radians(float(score_angle) * phase + rng.uniform(-float(config["score_angle_jitter_degrees"]), float(config["score_angle_jitter_degrees"])))
                magnitude = 1.0 + float(config["score_magnitude_step"]) * repetition
                scores.append(magnitude * np.asarray([math.cos(angle), math.sin(angle)]))
    clean_probability_array = np.asarray(clean_probability)
    group_ids_array = np.asarray(group_ids, dtype=int)
    context_ids_array = np.asarray(context_ids, dtype=int)
    true_channels = _channel_truth(context_count, channel_heterogeneity, config)
    fp = true_channels[context_ids_array, 0]
    fn = true_channels[context_ids_array, 1]
    observed_reward = fp + (1.0 - fp - fn) * clean_probability_array
    intervals = _audit_intervals(
        true_channels,
        interval_half_width,
        audit_size,
        rng,
        config,
        misspecified=misspecified,
    )
    return {
        "clean_probability": clean_probability_array,
        "clean_advantage": group_center(clean_probability_array, group_ids_array),
        "observed_reward": observed_reward,
        "group_ids": group_ids_array,
        "context_ids": context_ids_array,
        "scores": np.asarray(scores),
        "true_channels": true_channels,
        "channel_intervals": intervals,
    }


def _weights_metrics(
    weights: np.ndarray,
    true_advantage: np.ndarray,
    scores: np.ndarray,
    tolerance: float,
) -> dict[str, Any]:
    weights = np.asarray(weights, dtype=float)
    accepted = np.abs(weights) > tolerance
    clean_gradient = gradient_from_weights(true_advantage, scores)
    selected_gradient = gradient_from_weights(weights, scores)
    accepted_count = int(np.sum(accepted))
    false_signs = accepted & (weights * true_advantage <= tolerance)
    harmful = accepted & (weights * true_advantage < -tolerance)
    return {
        "selected_gradient": selected_gradient.tolist(),
        "clean_gradient_cosine": cosine(selected_gradient, clean_gradient, tolerance),
        "false_certified_sign_rate": float(np.sum(false_signs) / accepted_count) if accepted_count else 0.0,
        "harmful_sample_update_rate": float(np.sum(harmful) / accepted_count) if accepted_count else 0.0,
        "certified_mass": float(np.mean(accepted)),
        "accepted_count": accepted_count,
    }


def evaluate_problem(
    problem: dict[str, np.ndarray],
    seed: int,
    config: dict[str, Any],
    *,
    force_zero_certificate: bool = False,
) -> dict[str, Any]:
    tolerance = float(config["geometry_tolerance"])
    vertices, channel_corners = advantage_configurations(
        problem["observed_reward"],
        problem["context_ids"],
        problem["group_ids"],
        problem["channel_intervals"],
        tolerance,
    )
    intervals = advantage_intervals(vertices)
    if force_zero_certificate:
        intervals[:, 0] = np.minimum(intervals[:, 0], -1.0e-6)
        intervals[:, 1] = np.maximum(intervals[:, 1], 1.0e-6)
    h039_weights = certified_weights(intervals, tolerance)
    midpoint_channels = np.mean(problem["channel_intervals"], axis=2)
    point_clean = inverse_binary_channel(
        problem["observed_reward"], problem["context_ids"], midpoint_channels, tolerance
    )
    point_advantage = group_center(point_clean, problem["group_ids"])
    observed_advantage = group_center(problem["observed_reward"], problem["group_ids"])
    radii = 0.5 * (intervals[:, 1] - intervals[:, 0])
    accepted_count = int(np.sum(np.abs(h039_weights) > tolerance))

    ordered = np.lexsort((np.arange(len(radii)), radii))
    h010_weights = np.zeros_like(point_advantage)
    h010_weights[ordered[:accepted_count]] = point_advantage[ordered[:accepted_count]]
    random_order = np.random.default_rng(np.random.SeedSequence([seed, 39010])).permutation(len(radii))
    random_weights = np.zeros_like(point_advantage)
    random_weights[random_order[:accepted_count]] = point_advantage[random_order[:accepted_count]]

    parameter_radius = float(np.mean(radii))
    signcert_intervals = np.column_stack((point_advantage - parameter_radius, point_advantage + parameter_radius))
    signcert_weights = certified_weights(signcert_intervals, tolerance)
    scalar_radius = float(np.max(radii))
    scalar_weights = np.sign(point_advantage) * np.maximum(np.abs(point_advantage) - scalar_radius, 0.0)

    weights_by_method = {
        H039_METHOD: h039_weights,
        "observed_verifier_advantage": observed_advantage,
        "H001_point_channel_correction": point_advantage,
        "H010_uncertainty_mask": h010_weights,
        "SignCertPO_matched_parameter_radius": signcert_weights,
        "scalar_lower_bound_pessimism": scalar_weights,
        "matched_acceptance_random_filter": random_weights,
        "oracle_clean_advantage": problem["clean_advantage"],
    }
    method_metrics = {
        method: _weights_metrics(weights, problem["clean_advantage"], problem["scores"], tolerance)
        for method, weights in weights_by_method.items()
    }

    gradient_vertices = np.asarray([gradient_from_weights(vertex, problem["scores"]) for vertex in vertices])
    h027_direction, h027_abstained = minimum_norm_direction(gradient_vertices, tolerance)
    clean_gradient = gradient_from_weights(problem["clean_advantage"], problem["scores"])
    method_metrics["H027_global_gradient_set_direction"] = {
        "selected_gradient": h027_direction.tolist(),
        "clean_gradient_cosine": cosine(h027_direction, clean_gradient, tolerance),
        "false_certified_sign_rate": None,
        "harmful_sample_update_rate": None,
        "certified_mass": 0.0 if h027_abstained else 1.0,
        "accepted_count": 0 if h027_abstained else len(point_advantage),
    }

    nonoracle = [name for name in config["baselines"] if name != "oracle_clean_advantage"]
    best_cosine = max(float(method_metrics[name]["clean_gradient_cosine"]) for name in nonoracle)
    harmful_candidates = [
        float(method_metrics[name]["harmful_sample_update_rate"])
        for name in nonoracle
        if method_metrics[name]["harmful_sample_update_rate"] is not None
        and int(method_metrics[name]["accepted_count"]) >= accepted_count
    ]
    h039 = method_metrics[H039_METHOD]
    h001 = method_metrics["H001_point_channel_correction"]
    best_harmful = min(harmful_candidates) if harmful_candidates else 0.0
    h039["cosine_gain_over_best_nonoracle"] = float(h039["clean_gradient_cosine"]) - best_cosine
    h039["cosine_gain_over_H001"] = float(h039["clean_gradient_cosine"]) - float(h001["clean_gradient_cosine"])
    h039["harmful_rate_reduction_over_best_nonoracle"] = best_harmful - float(h039["harmful_sample_update_rate"])
    h039["harmful_rate_reduction_over_H001"] = float(h001["harmful_sample_update_rate"]) - float(h039["harmful_sample_update_rate"])

    truth_covered = bool(
        np.all(problem["true_channels"] >= problem["channel_intervals"][:, :, 0] - tolerance)
        and np.all(problem["true_channels"] <= problem["channel_intervals"][:, :, 1] + tolerance)
    )
    return {
        "method_metrics": method_metrics,
        "advantage_intervals": intervals.tolist(),
        "advantage_vertex_count": int(len(vertices)),
        "channel_corner_count": int(len(channel_corners)),
        "true_channel_covered": truth_covered,
        "matched_acceptance_count": accepted_count,
        "h010_acceptance_matches": method_metrics["H010_uncertainty_mask"]["accepted_count"] == accepted_count,
        "random_acceptance_matches": method_metrics["matched_acceptance_random_filter"]["accepted_count"] == accepted_count,
        "parameter_radius": parameter_radius,
        "channel_radius_mean": float(np.mean(radii)),
        "point_limit_direction_difference": float(
            np.linalg.norm(
                unit(np.asarray(h039["selected_gradient"]), tolerance)
                - unit(np.asarray(h001["selected_gradient"]), tolerance)
            )
        ),
    }


def _grid_row(
    seed: int,
    context_count: int,
    audit_size: int,
    interval_half_width: float,
    heterogeneity: float,
    score_angle: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    problem = make_problem(
        seed, context_count, audit_size, interval_half_width, heterogeneity, score_angle, config
    )
    evaluation = evaluate_problem(problem, seed, config)
    return {
        "seed": seed,
        "context_count": context_count,
        "audit_size": audit_size,
        "channel_interval_half_width": interval_half_width,
        "channel_heterogeneity": heterogeneity,
        "score_gradient_noncollinearity_angle": score_angle,
        **evaluation,
    }


def _control_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in config["seeds"]:
        for control in config["controls"]:
            width = 0.0 if control == "point_identified_channel" else 0.05
            heterogeneity = 0.0 if control == "symmetric_homogeneous_channel" else 0.15
            angle = 0.0 if control == "mixed_sign_cancellation" else 45.0
            misspecified = control == "channel_interval_misspecification"
            problem = make_problem(seed, 2, 128, width, heterogeneity, angle, config, misspecified=misspecified)
            evaluation = evaluate_problem(
                problem,
                seed,
                config,
                force_zero_certificate=control == "zero_certified_mass",
            )
            rows.append({"seed": seed, "control": control, **evaluation})
    return rows


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def run_experiment(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    coordinates = itertools.product(
        config["context_count_grid"],
        config["audit_size_grid"],
        config["channel_interval_half_width_grid"],
        config["channel_heterogeneity_grid"],
        config["score_gradient_noncollinearity_angle_grid"],
        config["seeds"],
    )
    rows = [_grid_row(seed, contexts, audit, width, heterogeneity, angle, config) for contexts, audit, width, heterogeneity, angle, seed in coordinates]
    coordinate_names = (
        "context_count",
        "audit_size",
        "channel_interval_half_width",
        "channel_heterogeneity",
        "score_gradient_noncollinearity_angle",
    )
    cells: list[dict[str, Any]] = []
    for key, grouped in itertools.groupby(
        sorted(rows, key=lambda row: tuple(row[name] for name in coordinate_names)),
        key=lambda row: tuple(row[name] for name in coordinate_names),
    ):
        subset = list(grouped)
        h039 = [row["method_metrics"][H039_METHOD] for row in subset]
        cell = dict(zip(coordinate_names, key))
        cell.update(
            {
                "false_certified_sign_rate": _mean(metric["false_certified_sign_rate"] for metric in h039),
                "harmful_sample_update_rate": _mean(metric["harmful_sample_update_rate"] for metric in h039),
                "clean_gradient_cosine": _mean(metric["clean_gradient_cosine"] for metric in h039),
                "certified_mass": _mean(metric["certified_mass"] for metric in h039),
                "cosine_gain_over_best_nonoracle": _mean(metric["cosine_gain_over_best_nonoracle"] for metric in h039),
                "cosine_gain_over_H001": _mean(metric["cosine_gain_over_H001"] for metric in h039),
                "harmful_rate_reduction_over_best_nonoracle": _mean(metric["harmful_rate_reduction_over_best_nonoracle"] for metric in h039),
                "harmful_rate_reduction_over_H001": _mean(metric["harmful_rate_reduction_over_H001"] for metric in h039),
                "point_limit_direction_difference": max(row["point_limit_direction_difference"] for row in subset),
                "valid_coverage": all(row["true_channel_covered"] for row in subset),
            }
        )
        cells.append(cell)

    thresholds = config["success_thresholds"]
    strong = [
        cell
        for cell in cells
        if cell["channel_interval_half_width"] <= float(config["strong_identification_interval_max"])
        and cell["audit_size"] >= int(config["strong_identification_audit_min"])
    ]
    gains = [
        cell
        for cell in cells
        if cell["channel_heterogeneity"] > 0.0
        and cell["score_gradient_noncollinearity_angle"] > 0.0
        and (
            cell["cosine_gain_over_H001"] >= float(thresholds["cosine_gain_min"])
            or cell["harmful_rate_reduction_over_H001"] >= float(thresholds["harmful_rate_reduction_min"])
        )
        and (
            cell["cosine_gain_over_best_nonoracle"] >= float(thresholds["cosine_gain_min"])
            or cell["harmful_rate_reduction_over_best_nonoracle"] >= float(thresholds["harmful_rate_reduction_min"])
        )
    ]
    point_controls = [row for row in _control_rows(config) if row["control"] == "point_identified_channel"]
    point_difference = max(row["point_limit_direction_difference"] for row in point_controls)
    passed = (
        max(cell["false_certified_sign_rate"] for cell in cells) <= float(thresholds["valid_coverage_false_certified_sign_rate_max"])
        and min(cell["clean_gradient_cosine"] for cell in strong) >= float(thresholds["strong_cell_clean_gradient_cosine_min"])
        and max(cell["harmful_sample_update_rate"] for cell in strong) <= float(thresholds["strong_cell_harmful_sample_update_rate_max"])
        and min(cell["certified_mass"] for cell in strong) >= float(thresholds["strong_cell_certified_mass_min"])
        and len(gains) >= int(thresholds["minimum_heterogeneous_noncollinear_gain_cells"])
        and point_difference <= float(thresholds["point_limit_direction_difference_max"])
    )
    return {
        "experiment_id": config["experiment_id"],
        "hypothesis_id": "H-039",
        "config_path": config_path.relative_to(ROOT).as_posix(),
        "seeds": config["seeds"],
        "language_model_training": False,
        "rows": rows,
        "control_rows": _control_rows(config),
        "summary": {
            "cells": cells,
            "valid_coverage_complete": all(cell["valid_coverage"] for cell in cells),
            "strong_identified_cell_count": len(strong),
            "qualifying_gain_cells": len(gains),
            "point_limit_direction_difference_max": point_difference,
        },
        "preregistered_outcome": "PASS" if passed else "FAIL",
    }


def write_results(payload: dict[str, Any], raw_path: Path, table_path: Path) -> None:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    cells = payload["summary"]["cells"]
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--raw-output", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--table-output", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--execute-formal-experiment", action="store_true")
    parser.add_argument("--confirmation-token")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not args.execute_formal_experiment or args.confirmation_token != config["formal_execution_confirmation_token"]:
        parser.error("formal execution requires the explicit flag and frozen confirmation token")
    if args.raw_output.exists() or args.table_output.exists():
        parser.error("formal H-039 result already exists; refusing to overwrite or rerun")
    payload = run_experiment(args.config)
    write_results(payload, args.raw_output, args.table_output)
    print(json.dumps({"experiment_id": payload["experiment_id"], "outcome": payload["preregistered_outcome"]}, indent=2))


if __name__ == "__main__":
    main()
