from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_HASH = "f07f69762e7e2b51d288feb1a76aebe9e7adcf4c8bf88fd626f82d70bef5f695"
EXPECTED_COMMIT = "90dbd2ea0db51d1a37adabb5678baab8e13bf24d"
EXPECTED_SEEDS = [11, 23, 47, 89, 131]
H033_METHOD = "privacy_stable_reusable_audit_gradient"


def close(actual: float, expected: float, tolerance: float = 1.0e-12) -> bool:
    return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance)


def main() -> None:
    errors: list[str] = []
    raw_path = ROOT / "07_results/raw/e0_h033_results.json"
    table_path = ROOT / "07_results/tables/e0_h033_summary.csv"
    hash_path = ROOT / "07_results/raw/e0_h033_results.sha256"
    card_path = ROOT / "07_results/result_cards/R-E0-H033.yaml"
    prereg_path = ROOT / "06_experiments/preregistrations/E0-H033.yaml"
    report_path = ROOT / "10_deliverables/h033_e0_experimental_report.md"
    review_path = ROOT / "08_reviews/local_reviews/h033_e0_review.md"
    for path in (raw_path, table_path, hash_path, card_path, prereg_path, report_path, review_path):
        if not path.is_file():
            errors.append(f"missing H-033 artifact: {path.relative_to(ROOT).as_posix()}")
    if errors:
        print(json.dumps({"errors": errors}, ensure_ascii=False, indent=2))
        sys.exit(1)

    raw_bytes = raw_path.read_bytes()
    canonical_bytes = raw_bytes.replace(b"\r\n", b"\n")
    actual_hash = hashlib.sha256(canonical_bytes).hexdigest()
    if actual_hash != EXPECTED_HASH:
        errors.append(f"raw result hash is {actual_hash} instead of {EXPECTED_HASH}")
    if not hash_path.read_text(encoding="utf-8").startswith(EXPECTED_HASH):
        errors.append("recorded SHA-256 does not match the frozen raw result")

    payload = json.loads(raw_bytes)
    if payload.get("experiment_id") != "E0-H033-PRIVACY-STABLE-REUSABLE-AUDIT-GRADIENT":
        errors.append("unexpected experiment ID")
    if payload.get("seeds") != EXPECTED_SEEDS:
        errors.append("formal seed list changed")
    if payload.get("language_model_training") is not False:
        errors.append("formal result does not declare language_model_training=false")
    if payload.get("preregistered_outcome") != "FAIL":
        errors.append("formal outcome is not FAIL")

    rows = payload.get("rows", [])
    controls = payload.get("control_rows", [])
    summary = payload.get("summary", {})
    cells = summary.get("cells", [])
    h033_cells = [cell for cell in cells if cell.get("method") == H033_METHOD]
    if len(rows) != 5760 or len(controls) != 35 or len(cells) != 1152 or len(h033_cells) != 144:
        errors.append("formal result does not preserve the 5760/35/1152/144 structure")
    if summary.get("qualifying_adaptive_gain_cells") != 0:
        errors.append("qualifying adaptive gain cell count changed")
    if summary.get("safety_pass") is not False or summary.get("round_efficiency_pass") is not False:
        errors.append("formal safety or round-efficiency failure changed")

    expected_extrema = {
        "population_gradient_bias_mean": (max, 3.0910191405872753),
        "clean_gradient_cosine_mean": (min, 0.019939108313956847),
        "false_positive_direction_rate": (max, 0.44880000000000003),
        "usable_round_ratio_over_disjoint_mean": (min, 1.0),
        "cosine_gain_over_naive_mean": (max, 0.0004950565415200892),
        "cosine_gain_over_best_exact_nonoracle_mean": (max, 0.003427226953351181),
    }
    for metric, (operation, expected) in expected_extrema.items():
        if h033_cells:
            actual = operation(float(cell[metric]) for cell in h033_cells)
            if not close(actual, expected):
                errors.append(f"H-033 {metric} extremum changed: {actual}")
    expected_failure_counts = {
        "bias": sum(float(cell["population_gradient_bias_mean"]) > 0.05 for cell in h033_cells),
        "cosine": sum(float(cell["clean_gradient_cosine_mean"]) < 0.90 for cell in h033_cells),
        "false_positive": sum(float(cell["false_positive_direction_rate"]) > 0.05 for cell in h033_cells),
        "round_ratio": sum(float(cell["usable_round_ratio_over_disjoint_mean"]) < 2.0 for cell in h033_cells),
    }
    if expected_failure_counts != {"bias": 118, "cosine": 127, "false_positive": 82, "round_ratio": 24}:
        errors.append(f"H-033 failure-cell counts changed: {expected_failure_counts}")

    h033_rows = [row for row in rows if row.get("method") == H033_METHOD]
    rho_error = max(
        (abs(float(row["privacy_rho_spent"]) - float(row["zcdp_rho_budget"])) for row in h033_rows),
        default=math.inf,
    )
    if rho_error > 1.0e-12:
        errors.append(f"privacy accountant exceeds tolerance: {rho_error}")
    privacy_off = [row for row in controls if row.get("control") == "privacy_off_limit"]
    if len(privacy_off) != 5 or any(
        not row.get("candidate_sequence_matches_naive")
        or not close(row.get("naive_bias_difference"), 0.0)
        or not close(row.get("naive_cosine_difference"), 0.0)
        for row in privacy_off
    ):
        errors.append("privacy-off control no longer matches naive reuse")
    clip_rows = [row for row in controls if row.get("control") == "contribution_clip_violation"]
    if len(clip_rows) != 5 or max(float(row["clipped_contribution_norm_max"]) for row in clip_rows) > 1.0 + 1.0e-12:
        errors.append("contribution clipping control exceeds tolerance")
    if len(table_path.read_text(encoding="utf-8").splitlines()) != 1153:
        errors.append("1152-cell summary table is missing or incomplete")

    result_card = yaml.safe_load(card_path.read_text(encoding="utf-8"))
    if result_card.get("decision") != "REJECTED_FAILED_PREDICTION_AND_DOMINATED":
        errors.append("result card decision is inconsistent")
    prereg = yaml.safe_load(prereg_path.read_text(encoding="utf-8"))
    if prereg.get("code_commit") != EXPECTED_COMMIT or prereg.get("status") != "COMPLETED_FAIL":
        errors.append("preregistration binding or terminal status is inconsistent")

    state = yaml.safe_load((ROOT / "research_state.yaml").read_text(encoding="utf-8"))
    active_ids = {path.stem for path in (ROOT / "05_hypotheses/active").glob("H-*.yaml")}
    if not {"H-001", "H-005", "H-014"} <= active_ids or "H-033" in active_ids:
        errors.append("current active portfolio does not reflect H-033 rejection")
    if set(state.get("branches", {}).get("active", [])) != active_ids:
        errors.append("research_state active portfolio differs from current cards")
    if state.get("budget", {}).get("used_units", 0) < 66:
        errors.append("current budget does not include the H-033 E0 unit")
    if (ROOT / "05_hypotheses/active/H-033.yaml").exists() or not (
        ROOT / "05_hypotheses/rejected/H-033.yaml"
    ).is_file():
        errors.append("H-033 was not moved from active to rejected")
    if len(active_ids) < 4 and not state.get("blockers"):
        errors.append("undersized active portfolio blocker is missing")

    lineage = json.loads((ROOT / "05_hypotheses/lineage_graph.json").read_text(encoding="utf-8"))
    h033_nodes = [node for node in lineage.get("nodes", []) if node.get("id") == "H-033"]
    if len(h033_nodes) != 1 or h033_nodes[0].get("status") != "REJECTED_FAILED_PREDICTION_AND_DOMINATED":
        errors.append("lineage graph does not record the H-033 terminal status")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    if "v0.11.3" not in readme or "3.091019" not in readme or "0.000495" not in readme:
        errors.append("README lacks the H-033 versioned result summary")
    decision_log = (ROOT / "09_decisions/decision_log.md").read_text(encoding="utf-8")
    if "D-0021" not in decision_log or "REJECT_H033_PRIVACY_UTILITY_AND_INCREMENTAL_GAIN" not in decision_log:
        errors.append("H-033 terminal decision is missing")

    output = {
        "experiment_id": payload.get("experiment_id"),
        "method_seed_rows": len(rows),
        "control_rows": len(controls),
        "summary_cells": len(cells),
        "h033_cells": len(h033_cells),
        "outcome": payload.get("preregistered_outcome"),
        "gain_cells": summary.get("qualifying_adaptive_gain_cells"),
        "raw_sha256": actual_hash,
        "errors": errors,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
