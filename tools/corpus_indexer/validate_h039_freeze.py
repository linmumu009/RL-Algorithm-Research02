from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_SEEDS = [11, 23, 47, 89, 131]
EXPECTED_BASELINES = {
    "observed_verifier_advantage",
    "H001_point_channel_correction",
    "H010_uncertainty_mask",
    "H027_global_gradient_set_direction",
    "SignCertPO_matched_parameter_radius",
    "scalar_lower_bound_pessimism",
    "matched_acceptance_random_filter",
    "oracle_clean_advantage",
}
EXPECTED_CONTROLS = {
    "point_identified_channel",
    "symmetric_homogeneous_channel",
    "mixed_sign_cancellation",
    "channel_interval_misspecification",
    "matched_parameter_vs_channel_radius",
    "equal_acceptance_rate",
    "zero_certified_mass",
    "equal_audit_and_compute",
}


def main() -> None:
    errors: list[str] = []
    config_path = ROOT / "06_experiments/configs/e0_h039.yaml"
    code_path = ROOT / "06_experiments/code/h039_sign_certificate_e0.py"
    test_path = ROOT / "06_experiments/unit_tests/test_h039_sign_certificate_e0.py"
    prereg_path = ROOT / "06_experiments/preregistrations/E0-H039.yaml"
    for path in (config_path, code_path, test_path, prereg_path):
        if not path.is_file() or path.stat().st_size < 500:
            errors.append(f"missing or short H-039 freeze artifact: {path.relative_to(ROOT).as_posix()}")

    raw_path = ROOT / "07_results/raw/e0_h039_results.json"
    table_path = ROOT / "07_results/tables/e0_h039_summary.csv"

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("experiment_id") != "E0-H039-CHANNEL-SET-ADVANTAGE-SIGN-CERTIFICATE":
        errors.append("unexpected H-039 experiment ID")
    if config.get("seeds") != EXPECTED_SEEDS:
        errors.append("H-039 seeds changed")
    expected_grids = {
        "context_count_grid": [2, 4],
        "audit_size_grid": [64, 128, 256],
        "channel_interval_half_width_grid": [0.02, 0.05, 0.10, 0.20],
        "channel_heterogeneity_grid": [0.00, 0.15, 0.30],
        "score_gradient_noncollinearity_angle_grid": [0, 45, 90],
    }
    for name, expected in expected_grids.items():
        if config.get(name) != expected:
            errors.append(f"H-039 frozen grid changed: {name}")
    aggregated_cells = 1
    for name in expected_grids:
        aggregated_cells *= len(config.get(name, []))
    if aggregated_cells != 216 or aggregated_cells * len(config.get("seeds", [])) != 1080:
        errors.append("H-039 grid is not 216 aggregated / 1080 seed cells")
    if set(config.get("baselines", [])) != EXPECTED_BASELINES:
        errors.append("H-039 comparator set changed")
    if set(config.get("controls", [])) != EXPECTED_CONTROLS:
        errors.append("H-039 control set changed")
    thresholds = config.get("success_thresholds", {})
    expected_thresholds = {
        "valid_coverage_false_certified_sign_rate_max": 0.05,
        "strong_cell_clean_gradient_cosine_min": 0.90,
        "strong_cell_harmful_sample_update_rate_max": 0.05,
        "strong_cell_certified_mass_min": 0.50,
        "cosine_gain_min": 0.05,
        "harmful_rate_reduction_min": 0.10,
        "minimum_heterogeneous_noncollinear_gain_cells": 3,
        "point_limit_direction_difference_max": 0.01,
    }
    if thresholds != expected_thresholds:
        errors.append("H-039 success thresholds changed")

    code = code_path.read_text(encoding="utf-8")
    required_code_tokens = [
        "enumerate_channel_corners",
        "group_center",
        "certified_weights",
        "H027_global_gradient_set_direction",
        "matched_acceptance_random_filter",
        "--execute-formal-experiment",
        "refusing to overwrite or rerun",
        'if __name__ == "__main__":',
    ]
    for token in required_code_tokens:
        if token not in code:
            errors.append(f"H-039 implementation lacks frozen token: {token}")

    prereg = yaml.safe_load(prereg_path.read_text(encoding="utf-8"))
    lifecycle = prereg.get("status")
    if lifecycle not in {"IMPLEMENTATION_FROZEN_NOT_RUN", "BOUND_NOT_RUN", "COMPLETED_FAIL"}:
        errors.append("H-039 preregistration is outside the frozen lifecycle")
    expected_commit = "TO_BE_SET_BEFORE_EXECUTION" if lifecycle == "IMPLEMENTATION_FROZEN_NOT_RUN" else "ddb66391e42bbaf5e63c85949df6c4fac8d32414"
    if prereg.get("code_commit") != expected_commit:
        errors.append("H-039 preregistration commit does not match its frozen lifecycle")
    if prereg.get("budget_limit") != 1:
        errors.append("H-039 budget changed")
    results_exist = raw_path.exists() and table_path.exists()
    if lifecycle in {"IMPLEMENTATION_FROZEN_NOT_RUN", "BOUND_NOT_RUN"} and (raw_path.exists() or table_path.exists()):
        errors.append("formal H-039 results exist before execution")
    if lifecycle == "COMPLETED_FAIL" and not results_exist:
        errors.append("completed H-039 lifecycle lacks formal results")

    state = yaml.safe_load((ROOT / "research_state.yaml").read_text(encoding="utf-8"))
    if state.get("budget", {}).get("used_units", 0) < 70:
        errors.append("research budget predates the H-039 freeze")
    latest_decision = str(state.get("latest_decision", {}).get("decision_id", ""))
    match = re.fullmatch(r"D-(\d{4})", latest_decision)
    if match is None or int(match.group(1)) < 23:
        errors.append("research_state predates the H-039 freeze")
    active = set(state.get("branches", {}).get("active", []))
    if not {"H-001", "H-005", "H-014"} <= active or (
        lifecycle in {"IMPLEMENTATION_FROZEN_NOT_RUN", "BOUND_NOT_RUN"} and "H-039" not in active
    ) or (lifecycle == "COMPLETED_FAIL" and "H-039" in active):
        errors.append("active portfolio is inconsistent with the H-039 lifecycle")

    decision_log = (ROOT / "09_decisions/decision_log.md").read_text(encoding="utf-8")
    if "D-0023" not in decision_log or "FREEZE_H039_E0_IMPLEMENTATION" not in decision_log:
        errors.append("H-039 freeze decision is missing")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    if "v0.12.1" not in readme or "1080 个 seed cell" not in readme or "34 项" not in readme:
        errors.append("README lacks the H-039 freeze summary")

    output = {
        "experiment_id": config.get("experiment_id"),
        "aggregated_cells": aggregated_cells,
        "seed_cells": aggregated_cells * len(config.get("seeds", [])),
        "baselines": len(config.get("baselines", [])),
        "controls": len(config.get("controls", [])),
        "formal_results_absent": not raw_path.exists() and not table_path.exists(),
        "errors": errors,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
