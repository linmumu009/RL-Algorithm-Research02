from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_HASH = "9df09d3e5f5a837b40dbcb5af1b2e89f131806258932899b2a2490cffa5792bf"
EXPECTED_COMMIT = "ddb66391e42bbaf5e63c85949df6c4fac8d32414"
EXPECTED_SEEDS = [11, 23, 47, 89, 131]
H039_METHOD = "channel_set_advantage_sign_certificate"


def close(actual: float, expected: float, tolerance: float = 1.0e-12) -> bool:
    return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance)


def main() -> None:
    errors: list[str] = []
    raw_path = ROOT / "07_results/raw/e0_h039_results.json"
    table_path = ROOT / "07_results/tables/e0_h039_summary.csv"
    hash_path = ROOT / "07_results/raw/e0_h039_results.sha256"
    card_path = ROOT / "07_results/result_cards/R-E0-H039.yaml"
    prereg_path = ROOT / "06_experiments/preregistrations/E0-H039.yaml"
    report_path = ROOT / "10_deliverables/h039_e0_experimental_report.md"
    review_path = ROOT / "08_reviews/local_reviews/h039_e0_review.md"
    for path in (raw_path, table_path, hash_path, card_path, prereg_path, report_path, review_path):
        if not path.is_file():
            errors.append(f"missing H-039 artifact: {path.relative_to(ROOT).as_posix()}")
    if errors:
        print(json.dumps({"errors": errors}, ensure_ascii=False, indent=2))
        sys.exit(1)

    raw_bytes = raw_path.read_bytes()
    canonical = raw_bytes.replace(b"\r\n", b"\n")
    actual_hash = hashlib.sha256(canonical).hexdigest()
    if actual_hash != EXPECTED_HASH:
        errors.append(f"raw result hash is {actual_hash} instead of {EXPECTED_HASH}")
    if not hash_path.read_text(encoding="utf-8").startswith(EXPECTED_HASH):
        errors.append("recorded SHA-256 does not match the frozen raw result")

    payload = json.loads(raw_bytes)
    if payload.get("experiment_id") != "E0-H039-CHANNEL-SET-ADVANTAGE-SIGN-CERTIFICATE":
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
    if len(rows) != 1080 or len(controls) != 40 or len(cells) != 216:
        errors.append("formal result does not preserve the 1080/40/216 structure")
    if summary.get("valid_coverage_complete") is not True or summary.get("strong_identified_cell_count") != 72:
        errors.append("coverage or strong-cell structure changed")
    if summary.get("qualifying_gain_cells") != 0:
        errors.append("qualifying gain cell count changed")
    if not close(summary.get("point_limit_direction_difference_max"), 0.0):
        errors.append("point-limit equality changed")

    target = [
        cell for cell in cells
        if float(cell["channel_heterogeneity"]) > 0.0
        and float(cell["score_gradient_noncollinearity_angle"]) > 0.0
    ]
    strong = [
        cell for cell in cells
        if float(cell["channel_interval_half_width"]) <= 0.05 and int(cell["audit_size"]) >= 128
    ]
    expected = {
        "valid_false_max": (max(float(cell["false_certified_sign_rate"]) for cell in cells), 0.0),
        "valid_harmful_max": (max(float(cell["harmful_sample_update_rate"]) for cell in cells), 0.0),
        "strong_cosine_min": (min(float(cell["clean_gradient_cosine"]) for cell in strong), 0.9997177274787099),
        "strong_mass_min": (min(float(cell["certified_mass"]) for cell in strong), 2.0 / 3.0),
        "target_cosine_gain_max": (max(float(cell["cosine_gain_over_best_nonoracle"]) for cell in target), 1.2790766977577305e-06),
        "target_harmful_reduction_max": (max(float(cell["harmful_rate_reduction_over_best_nonoracle"]) for cell in target), 0.0),
        "all_cosine_min": (min(float(cell["clean_gradient_cosine"]) for cell in cells), -0.2184338166724597),
    }
    for name, (actual, wanted) in expected.items():
        if not close(actual, wanted):
            errors.append(f"H-039 {name} changed: {actual}")
    if len(target) != 96:
        errors.append("target cell count is not 96")
    if sum(float(cell["certified_mass"]) == 0.0 for cell in cells) != 15:
        errors.append("zero-certified-mass cell count changed")
    if sum(float(cell["clean_gradient_cosine"]) < 0.0 for cell in cells) != 4:
        errors.append("negative-cosine cell count changed")

    if any(not row.get("h010_acceptance_matches") or not row.get("random_acceptance_matches") for row in rows):
        errors.append("matched-acceptance invariant changed")
    if max(abs(float(row["parameter_radius"]) - float(row["channel_radius_mean"])) for row in rows) > 1.0e-12:
        errors.append("matched-radius invariant changed")
    point_controls = [row for row in controls if row.get("control") == "point_identified_channel"]
    zero_controls = [row for row in controls if row.get("control") == "zero_certified_mass"]
    if len(point_controls) != 5 or any(not close(row["point_limit_direction_difference"], 0.0) for row in point_controls):
        errors.append("point-identified control changed")
    if len(zero_controls) != 5 or any(not close(row["method_metrics"][H039_METHOD]["certified_mass"], 0.0) for row in zero_controls):
        errors.append("zero-certificate control changed")
    if len(table_path.read_text(encoding="utf-8").splitlines()) != 217:
        errors.append("216-cell summary table is incomplete")

    card = yaml.safe_load(card_path.read_text(encoding="utf-8"))
    if card.get("decision") != "REJECTED_NO_INCREMENTAL_GAIN_AND_DOMINATED":
        errors.append("result card decision is inconsistent")
    prereg = yaml.safe_load(prereg_path.read_text(encoding="utf-8"))
    if prereg.get("code_commit") != EXPECTED_COMMIT or prereg.get("status") != "COMPLETED_FAIL":
        errors.append("preregistration binding or terminal status is inconsistent")
    state = yaml.safe_load((ROOT / "research_state.yaml").read_text(encoding="utf-8"))
    current = set(state.get("branches", {}).get("active", [])) | set(state.get("branches", {}).get("paused", []))
    if current != {"H-001", "H-005", "H-014"}:
        errors.append("current portfolio does not reflect H-039 rejection")
    latest_decision = str(state.get("latest_decision", {}).get("decision_id", ""))
    if state.get("budget", {}).get("used_units", 0) < 71 or not latest_decision.startswith("D-") or int(latest_decision[2:]) < 25:
        errors.append("terminal budget or decision predates H-039 rejection")
    if len(state.get("blockers", [])) < 2:
        errors.append("portfolio/reserve blockers are missing")
    lineage = json.loads((ROOT / "05_hypotheses/lineage_graph.json").read_text(encoding="utf-8"))
    nodes = [node for node in lineage.get("nodes", []) if node.get("id") == "H-039"]
    if len(nodes) != 1 or nodes[0].get("status") != "REJECTED_NO_INCREMENTAL_GAIN_AND_DOMINATED":
        errors.append("lineage graph does not record H-039 rejection")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    if "v0.12.3" not in readme or "0.000001279" not in readme or "71/100" not in readme:
        errors.append("README lacks the H-039 terminal summary")

    output = {
        "experiment_id": payload.get("experiment_id"),
        "seed_rows": len(rows),
        "control_rows": len(controls),
        "summary_cells": len(cells),
        "strong_cells": len(strong),
        "target_cells": len(target),
        "outcome": payload.get("preregistered_outcome"),
        "gain_cells": summary.get("qualifying_gain_cells"),
        "raw_sha256": actual_hash,
        "errors": errors,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
