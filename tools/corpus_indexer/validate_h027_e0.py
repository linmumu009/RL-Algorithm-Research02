from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_HASH = "8bbf6df31fbb17a40e3bdd723dd9edc70343da074ffe78b6f194ede1ef0190ac"
EXPECTED_COMMIT = "b2778d683b22d8f7a24f60e3d3443abb2671aed2"
EXPECTED_SEEDS = [11, 23, 47, 89, 131]


def close(actual: float, expected: float, tolerance: float = 1.0e-12) -> bool:
    return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance)


def main() -> None:
    errors: list[str] = []
    raw_path = ROOT / "07_results/raw/e0_h027_results.json"
    table_path = ROOT / "07_results/tables/e0_h027_summary.csv"
    hash_path = ROOT / "07_results/raw/e0_h027_results.sha256"
    card_path = ROOT / "07_results/result_cards/R-E0-H027.yaml"
    prereg_path = ROOT / "06_experiments/preregistrations/E0-H027.yaml"
    for path in (raw_path, table_path, hash_path, card_path, prereg_path):
        if not path.is_file():
            errors.append(f"missing H-027 artifact: {path.relative_to(ROOT).as_posix()}")
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
    if payload.get("experiment_id") != "E0-H027-AUDIT-IDENTIFIED-GRADIENT-SET":
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
    if len(rows) != 400 or len(controls) != 30 or len(cells) != 80:
        errors.append("formal result does not preserve the 400/30/80 row structure")
    expected_summary = {
        "valid_coverage_complete": True,
        "valid_cell_false_positive_direction_rate_max": 0.0,
        "strong_identified_cell_count": 64,
        "asymmetric_noncollinear_gain_cells": 0,
        "zero_in_set_abstention_rate": 1.0,
        "point_limit_direction_difference_max": 0.0,
        "misspecified_interval_harmful_direction_rate": 1.0,
    }
    for key, expected in expected_summary.items():
        actual = summary.get(key)
        if isinstance(expected, float):
            if actual is None or not close(actual, expected):
                errors.append(f"summary {key} is {actual!r} instead of {expected!r}")
        elif actual != expected:
            errors.append(f"summary {key} is {actual!r} instead of {expected!r}")
    if not close(summary.get("strong_identified_clean_gradient_cosine_min"), 0.995152974264605):
        errors.append("strong identified cosine minimum changed")
    if len(table_path.read_text(encoding="utf-8").splitlines()) != 81:
        errors.append("80-cell summary table is missing or incomplete")

    result_card = yaml.safe_load(card_path.read_text(encoding="utf-8"))
    if result_card.get("decision") != "REJECTED_NO_INCREMENTAL_GAIN_AND_DOMINATED":
        errors.append("result card decision is inconsistent")
    prereg = yaml.safe_load(prereg_path.read_text(encoding="utf-8"))
    if prereg.get("code_commit") != EXPECTED_COMMIT or prereg.get("status") != "COMPLETED_FAIL":
        errors.append("preregistration binding or terminal status is inconsistent")

    state = yaml.safe_load((ROOT / "research_state.yaml").read_text(encoding="utf-8"))
    if set(state.get("branches", {}).get("active", [])) != {"H-001", "H-005", "H-014"}:
        errors.append("current active portfolio does not reflect H-027 rejection")
    if state.get("budget", {}).get("used_units") != 61:
        errors.append("current budget does not include the H-027 E0 unit")
    if (ROOT / "05_hypotheses/active/H-027.yaml").exists() or not (ROOT / "05_hypotheses/rejected/H-027.yaml").is_file():
        errors.append("H-027 was not moved from active to rejected")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    if "v0.10.0" not in readme or "0.011294" not in readme:
        errors.append("README lacks the H-027 versioned result summary")
    decision_log = (ROOT / "09_decisions/decision_log.md").read_text(encoding="utf-8")
    if "D-0017" not in decision_log or "REJECT_H027_NO_INCREMENTAL_GAIN" not in decision_log:
        errors.append("H-027 terminal decision is missing")

    output = {
        "experiment_id": payload.get("experiment_id"),
        "raw_rows": len(rows),
        "control_rows": len(controls),
        "valid_cells": len(cells),
        "outcome": payload.get("preregistered_outcome"),
        "gain_cells": summary.get("asymmetric_noncollinear_gain_cells"),
        "raw_sha256": actual_hash,
        "errors": errors,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
