"""Frozen E0 for H-027: audit-identified clean-gradient sets.

The formal experiment is analytic/seeded synthetic and two-dimensional. It
maps fixed false-positive/false-negative audit intervals to the exact four
feasible clean-gradient vertices, projects zero onto their convex hull, and
uses the normalized minimum-norm point only when the hull is separated from
zero. Importing this module never executes or writes the formal experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "06_experiments" / "configs" / "e0_h027.yaml"
DEFAULT_RAW = ROOT / "07_results" / "raw" / "e0_h027_results.json"
DEFAULT_TABLE = ROOT / "07_results" / "tables" / "e0_h027_summary.csv"


def unit(vector: np.ndarray, tolerance: float = 1.0e-12) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= tolerance:
        return np.zeros_like(vector)
    return vector / norm


def cosine(left: np.ndarray, right: np.ndarray, tolerance: float = 1.0e-12) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= tolerance or right_norm <= tolerance:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def rotate(vector: np.ndarray, degrees: float) -> np.ndarray:
    radians = math.radians(float(degrees))
    matrix = np.asarray(
        [[math.cos(radians), -math.sin(radians)], [math.sin(radians), math.cos(radians)]],
        dtype=float,
    )
    return matrix @ np.asarray(vector, dtype=float)


def _cross(origin: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    left_delta = left - origin
    right_delta = right - origin
    return float(left_delta[0] * right_delta[1] - left_delta[1] * right_delta[0])


def convex_hull(points: Sequence[Sequence[float]], tolerance: float = 1.0e-12) -> np.ndarray:
    """Return unique two-dimensional hull vertices in counter-clockwise order."""

    unique = sorted({(float(point[0]), float(point[1])) for point in points})
    if len(unique) <= 1:
        return np.asarray(unique, dtype=float)
    arrays = [np.asarray(point, dtype=float) for point in unique]
    lower: list[np.ndarray] = []
    for point in arrays:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= tolerance:
            lower.pop()
        lower.append(point)
    upper: list[np.ndarray] = []
    for point in reversed(arrays):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= tolerance:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    return np.asarray(hull, dtype=float)


def point_in_convex_hull(point: np.ndarray, hull: np.ndarray, tolerance: float = 1.0e-12) -> bool:
    point = np.asarray(point, dtype=float)
    hull = np.asarray(hull, dtype=float)
    if len(hull) == 0:
        return False
    if len(hull) == 1:
        return bool(np.linalg.norm(point - hull[0]) <= tolerance)
    if len(hull) == 2:
        segment = hull[1] - hull[0]
        denominator = float(np.dot(segment, segment))
        if denominator <= tolerance:
            return bool(np.linalg.norm(point - hull[0]) <= tolerance)
        coefficient = float(np.dot(point - hull[0], segment) / denominator)
        projection = hull[0] + np.clip(coefficient, 0.0, 1.0) * segment
        return bool(np.linalg.norm(point - projection) <= tolerance)
    signs = [_cross(hull[index], hull[(index + 1) % len(hull)], point) for index in range(len(hull))]
    return all(value >= -tolerance for value in signs) or all(value <= tolerance for value in signs)


def _closest_on_segment(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    segment = end - start
    denominator = float(np.dot(segment, segment))
    if denominator == 0.0:
        return start.copy()
    coefficient = float(np.clip(-np.dot(start, segment) / denominator, 0.0, 1.0))
    return start + coefficient * segment


def minimum_norm_point(points: Sequence[Sequence[float]], tolerance: float = 1.0e-12) -> tuple[np.ndarray, np.ndarray]:
    """Project zero onto a 2-D convex hull without an external optimizer."""

    hull = convex_hull(points, tolerance)
    if len(hull) == 0:
        raise ValueError("At least one feasible gradient vertex is required")
    origin = np.zeros(2, dtype=float)
    if point_in_convex_hull(origin, hull, tolerance):
        return origin, hull
    if len(hull) == 1:
        return hull[0].copy(), hull
    candidates = [
        _closest_on_segment(hull[index], hull[(index + 1) % len(hull)])
        for index in range(len(hull))
    ]
    return min(candidates, key=lambda value: float(np.dot(value, value))).copy(), hull


def maximin_direction(
    points: Sequence[Sequence[float]], tolerance: float = 1.0e-12
) -> dict[str, Any]:
    minimum, hull = minimum_norm_point(points, tolerance)
    norm = float(np.linalg.norm(minimum))
    if norm <= tolerance:
        return {
            "direction": np.zeros(2, dtype=float),
            "minimum_norm_point": minimum,
            "hull": hull,
            "worst_case_alignment_margin": 0.0,
            "abstained": True,
        }
    direction = minimum / norm
    margin = float(np.min(hull @ direction))
    return {
        "direction": direction,
        "minimum_norm_point": minimum,
        "hull": hull,
        "worst_case_alignment_margin": margin,
        "abstained": margin <= tolerance,
    }


def _seeded_clean_gradient(seed: int, config: dict[str, Any]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    angle = float(config["clean_gradient_base_angle_degrees"]) + rng.uniform(
        -float(config["seed_angle_jitter_degrees"]),
        float(config["seed_angle_jitter_degrees"]),
    )
    magnitude = float(config["clean_gradient_magnitude"]) * (
        1.0 + rng.uniform(-float(config["seed_magnitude_jitter"]), float(config["seed_magnitude_jitter"]))
    )
    return magnitude * rotate(np.asarray([1.0, 0.0]), angle)


def audit_compatible_vertices(
    clean_gradient: np.ndarray,
    interval_half_width: float,
    channel_asymmetry: float,
    score_gradient_angle: float,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Map two fixed channel intervals to their exact four gradient vertices.

    Zero is the true channel coordinate. Asymmetry moves the interval midpoint
    while preserving truth coverage; at maximum asymmetry truth lies on the
    lower corner. The two axes represent score-weighted FP and FN moments.
    """

    clean_unit = unit(clean_gradient, float(config["geometry_tolerance"]))
    maximum_asymmetry = float(config["maximum_asymmetry"])
    ratio = 0.0 if maximum_asymmetry == 0.0 else float(channel_asymmetry) / maximum_asymmetry
    lower = -float(interval_half_width) * (1.0 - ratio)
    upper = float(interval_half_width) * (1.0 + ratio)
    false_positive_axis = float(config["false_positive_axis_scale"]) * rotate(
        clean_unit, float(score_gradient_angle)
    )
    false_negative_axis = float(config["false_negative_axis_scale"]) * rotate(
        clean_unit, -0.5 * float(score_gradient_angle)
    )
    vertices = np.asarray(
        [
            clean_gradient + fp * false_positive_axis + fn * false_negative_axis
            for fp in (lower, upper)
            for fn in (lower, upper)
        ],
        dtype=float,
    )
    return vertices, {
        "false_positive_axis": false_positive_axis,
        "false_negative_axis": false_negative_axis,
        "channel_lower": np.asarray([lower, lower], dtype=float),
        "channel_upper": np.asarray([upper, upper], dtype=float),
    }


def _diameter(points: np.ndarray) -> float:
    return max(float(np.linalg.norm(left - right)) for left in points for right in points)


def _direction_metrics(
    clean_gradient: np.ndarray,
    vertices: np.ndarray,
    axes: dict[str, np.ndarray],
    interval_half_width: float,
    channel_asymmetry: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    tolerance = float(config["geometry_tolerance"])
    robust = maximin_direction(vertices, tolerance)
    direction = robust["direction"]
    midpoint = np.mean(vertices, axis=0)
    asymmetry_ratio = (
        0.0
        if float(config["maximum_asymmetry"]) == 0.0
        else float(channel_asymmetry) / float(config["maximum_asymmetry"])
    )
    observed = clean_gradient + float(config["observed_bias_scale"]) * asymmetry_ratio * float(
        interval_half_width
    ) * (axes["false_positive_axis"] + axes["false_negative_axis"])
    h001 = unit(midpoint, tolerance)
    h018 = unit(0.75 * observed, tolerance)
    kl_dro = unit(math.exp(-float(interval_half_width)) * observed, tolerance)
    norm_matched = unit(h001 * float(np.linalg.norm(direction)), tolerance)
    baseline_directions = {
        "H001_midpoint_channel_correction": h001,
        "H018_scalar_lower_bound_reward": h018,
        "kl_dro_scalar_pessimism": kl_dro,
        "norm_matched_gradient_shrinkage": norm_matched,
    }
    baseline_cosines = {
        name: cosine(candidate, clean_gradient, tolerance) for name, candidate in baseline_directions.items()
    }
    best_name = max(baseline_cosines, key=baseline_cosines.get)
    robust_cosine = cosine(direction, clean_gradient, tolerance)
    harmful = bool(not robust["abstained"] and np.dot(direction, clean_gradient) <= 0.0)
    return {
        "direction": direction,
        "minimum_norm_point": robust["minimum_norm_point"],
        "hull": robust["hull"],
        "worst_case_alignment_margin": float(robust["worst_case_alignment_margin"]),
        "abstained": bool(robust["abstained"]),
        "clean_gradient_cosine": robust_cosine,
        "cosine_gain_over_best_nonoracle": robust_cosine - baseline_cosines[best_name],
        "best_nonoracle": best_name,
        "best_nonoracle_cosine": baseline_cosines[best_name],
        "baseline_cosines": baseline_cosines,
        "false_positive_direction": harmful,
        "oracle_regret": 1.0 - robust_cosine if not robust["abstained"] else 1.0,
        "identified_set_diameter": _diameter(vertices),
        "point_direction_difference": float(np.linalg.norm(direction - h001)),
    }


def _json_vector(value: np.ndarray) -> list[float]:
    return [float(item) for item in np.asarray(value, dtype=float)]


def _grid_row(
    seed: int,
    interval_half_width: float,
    channel_asymmetry: float,
    score_gradient_angle: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    clean_gradient = _seeded_clean_gradient(seed, config)
    vertices, axes = audit_compatible_vertices(
        clean_gradient, interval_half_width, channel_asymmetry, score_gradient_angle, config
    )
    metrics = _direction_metrics(
        clean_gradient, vertices, axes, interval_half_width, channel_asymmetry, config
    )
    return {
        "row_type": "valid_coverage_grid",
        "seed": seed,
        "interval_half_width": interval_half_width,
        "channel_asymmetry": channel_asymmetry,
        "score_gradient_angle": score_gradient_angle,
        "true_channel_covered": point_in_convex_hull(
            clean_gradient, convex_hull(vertices, float(config["geometry_tolerance"])), float(config["geometry_tolerance"])
        ),
        "clean_gradient": _json_vector(clean_gradient),
        "feasible_vertices": [_json_vector(vertex) for vertex in vertices],
        "minimum_norm_point": _json_vector(metrics.pop("minimum_norm_point")),
        "selected_direction": _json_vector(metrics.pop("direction")),
        "convex_hull": [_json_vector(vertex) for vertex in metrics.pop("hull")],
        **metrics,
    }


def _control_row(seed: int, control: str, config: dict[str, Any]) -> dict[str, Any]:
    clean = _seeded_clean_gradient(seed, config)
    tolerance = float(config["geometry_tolerance"])
    if control == "point_identified_channel":
        vertices, axes = audit_compatible_vertices(clean, 0.0, 0.0, 60.0, config)
        true_covered = True
    elif control == "symmetric_zero_bias":
        vertices, axes = audit_compatible_vertices(clean, 0.10, 0.0, 60.0, config)
        true_covered = True
    elif control in {"zero_in_identified_set", "wide_interval_abstention"}:
        width = 1.25 if control == "zero_in_identified_set" else float(config["wide_interval_half_width"])
        basis = unit(clean, tolerance)
        perpendicular = rotate(basis, 90.0)
        vertices = np.asarray(
            [clean + left * basis + right * perpendicular for left in (-width, width) for right in (-width, width)]
        )
        axes = {"false_positive_axis": basis, "false_negative_axis": perpendicular}
        true_covered = True
    elif control == "misspecified_interval_excludes_truth":
        basis = unit(clean, tolerance)
        perpendicular = rotate(basis, 90.0)
        center = -0.75 * clean
        vertices = np.asarray(
            [center + left * basis + right * perpendicular for left in (-0.05, 0.05) for right in (-0.05, 0.05)]
        )
        axes = {"false_positive_axis": basis, "false_negative_axis": perpendicular}
        true_covered = False
    elif control == "equal_compute_and_audit_count":
        vertices, axes = audit_compatible_vertices(clean, 0.10, 0.24, 60.0, config)
        true_covered = True
    else:
        raise ValueError(f"Unknown control: {control}")
    metrics = _direction_metrics(clean, vertices, axes, 0.10, 0.24, config)
    return {
        "row_type": "control",
        "control": control,
        "seed": seed,
        "true_channel_covered": true_covered,
        "clean_gradient": _json_vector(clean),
        "feasible_vertices": [_json_vector(vertex) for vertex in vertices],
        "minimum_norm_point": _json_vector(metrics.pop("minimum_norm_point")),
        "selected_direction": _json_vector(metrics.pop("direction")),
        "convex_hull": [_json_vector(vertex) for vertex in metrics.pop("hull")],
        **metrics,
    }


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def run_experiment(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in config["seeds"]]
    widths = [float(value) for value in config["interval_half_width_grid"]]
    asymmetries = [float(value) for value in config["channel_asymmetry_grid"]]
    angles = [float(value) for value in config["score_gradient_angle_grid"]]
    rows = [
        _grid_row(seed, width, asymmetry, angle, config)
        for width in widths
        for asymmetry in asymmetries
        for angle in angles
        for seed in seeds
    ]
    controls = [
        "point_identified_channel",
        "symmetric_zero_bias",
        "zero_in_identified_set",
        "misspecified_interval_excludes_truth",
        "wide_interval_abstention",
        "equal_compute_and_audit_count",
    ]
    control_rows = [_control_row(seed, control, config) for control in controls for seed in seeds]

    cells: list[dict[str, Any]] = []
    for width in widths:
        for asymmetry in asymmetries:
            for angle in angles:
                subset = [
                    row
                    for row in rows
                    if row["interval_half_width"] == width
                    and row["channel_asymmetry"] == asymmetry
                    and row["score_gradient_angle"] == angle
                ]
                cells.append(
                    {
                        "interval_half_width": width,
                        "channel_asymmetry": asymmetry,
                        "score_gradient_angle": angle,
                        "clean_gradient_cosine_mean": _mean(row["clean_gradient_cosine"] for row in subset),
                        "cosine_gain_over_best_nonoracle_mean": _mean(
                            row["cosine_gain_over_best_nonoracle"] for row in subset
                        ),
                        "false_positive_direction_rate": _mean(
                            float(row["false_positive_direction"]) for row in subset
                        ),
                        "abstention_rate": _mean(float(row["abstained"]) for row in subset),
                        "worst_case_alignment_margin_mean": _mean(
                            row["worst_case_alignment_margin"] for row in subset
                        ),
                        "identified_set_diameter_mean": _mean(row["identified_set_diameter"] for row in subset),
                        "point_direction_difference_max": max(row["point_direction_difference"] for row in subset),
                    }
                )

    thresholds = config["success_thresholds"]
    valid_max_false_positive = max(cell["false_positive_direction_rate"] for cell in cells)
    strong_cells = [
        cell
        for cell in cells
        if cell["interval_half_width"] <= float(config["strong_identification_width_max"])
        and cell["abstention_rate"] == 0.0
    ]
    strong_min_cosine = min(cell["clean_gradient_cosine_mean"] for cell in strong_cells)
    gain_cells = [
        cell
        for cell in cells
        if cell["channel_asymmetry"] > 0.0
        and cell["score_gradient_angle"] not in (0.0, 180.0)
        and cell["cosine_gain_over_best_nonoracle_mean"] >= float(thresholds["cosine_gain_min"])
    ]
    zero_rows = [row for row in control_rows if row["control"] == "zero_in_identified_set"]
    misspecified_rows = [row for row in control_rows if row["control"] == "misspecified_interval_excludes_truth"]
    point_cells = [cell for cell in cells if cell["interval_half_width"] == 0.0]
    zero_abstention_rate = _mean(float(row["abstained"]) for row in zero_rows)
    point_limit_difference = max(cell["point_direction_difference_max"] for cell in point_cells)
    misspecified_harmful_rate = _mean(float(row["false_positive_direction"]) for row in misspecified_rows)
    coverage_complete = all(row["true_channel_covered"] for row in rows)
    passed = (
        coverage_complete
        and valid_max_false_positive <= float(thresholds["valid_cell_false_positive_direction_rate_max"])
        and strong_min_cosine >= float(thresholds["strong_identified_clean_gradient_cosine_min"])
        and len(gain_cells) >= int(thresholds["minimum_asymmetric_noncollinear_gain_cells"])
        and zero_abstention_rate >= float(thresholds["zero_in_set_abstention_rate_min"])
        and point_limit_difference <= float(thresholds["point_limit_direction_difference_max"])
    )
    return {
        "experiment_id": config["experiment_id"],
        "hypothesis_id": "H-027",
        "config_path": config_path.relative_to(ROOT).as_posix(),
        "seeds": seeds,
        "language_model_training": False,
        "rows": rows,
        "control_rows": control_rows,
        "summary": {
            "cells": cells,
            "valid_coverage_complete": coverage_complete,
            "valid_cell_false_positive_direction_rate_max": valid_max_false_positive,
            "strong_identified_cell_count": len(strong_cells),
            "strong_identified_clean_gradient_cosine_min": strong_min_cosine,
            "asymmetric_noncollinear_gain_cells": len(gain_cells),
            "gain_cell_coordinates": [
                [cell["interval_half_width"], cell["channel_asymmetry"], cell["score_gradient_angle"]]
                for cell in gain_cells
            ],
            "zero_in_set_abstention_rate": zero_abstention_rate,
            "point_limit_direction_difference_max": point_limit_difference,
            "misspecified_interval_harmful_direction_rate": misspecified_harmful_rate,
        },
        "preregistered_outcome": "PASS" if passed else "FAIL",
    }


def write_results(payload: dict[str, Any], raw_path: Path, table_path: Path) -> None:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    fieldnames = [
        "interval_half_width",
        "channel_asymmetry",
        "score_gradient_angle",
        "clean_gradient_cosine_mean",
        "cosine_gain_over_best_nonoracle_mean",
        "false_positive_direction_rate",
        "abstention_rate",
        "worst_case_alignment_margin_mean",
        "identified_set_diameter_mean",
        "point_direction_difference_max",
    ]
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for cell in payload["summary"]["cells"]:
            writer.writerow({name: cell[name] for name in fieldnames})


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
