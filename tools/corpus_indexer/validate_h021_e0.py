from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_HASH = "cba8540ab7570e874e99b9adb50ddc6b85a47ae78e5b0d217d6724014105896b"
EXPECTED_SEEDS = [11, 23, 47, 89, 131]


def main() -> None:
    errors: list[str] = []
    raw_path = ROOT / "07_results/raw/e0_h021_results.json"
    table_path = ROOT / "07_results/tables/e0_h021_summary.csv"
    card_path = ROOT / "07_results/result_cards/R-E0-H021.yaml"
    prereg_path = ROOT / "06_experiments/preregistrations/E0-H021.yaml"

    raw_bytes = raw_path.read_bytes()
    canonical_bytes = raw_bytes.replace(b"\r\n", b"\n")
    actual_hash = hashlib.sha256(canonical_bytes).hexdigest()
    if actual_hash != EXPECTED_HASH:
        errors.append(f"raw result hash is {actual_hash} instead of {EXPECTED_HASH}")
    payload = json.loads(raw_bytes)
    if payload.get("experiment_id") != "E0-H021-NEGATIVE-CONTROL-BRIDGE":
        errors.append("unexpected experiment ID")
    if payload.get("seeds") != EXPECTED_SEEDS:
        errors.append("formal seed list changed")
    if payload.get("language_model_training") is not False:
        errors.append("formal result does not declare language_model_training=false")
    if payload.get("preregistered_outcome") != "FAIL":
        errors.append("formal outcome is not FAIL")
    rows = payload.get("rows", [])
    if len(rows) != 95:
        errors.append(f"raw result contains {len(rows)} rows instead of 95")
    valid = [row for row in rows if not row.get("invalid_exclusion")]
    invalid = [row for row in rows if row.get("invalid_exclusion")]
    if len(valid) != 90 or len(invalid) != 5:
        errors.append("valid/invalid-exclusion row split is not 90/5")

    summary = payload.get("summary", {})
    if len(summary.get("cells", [])) != 18:
        errors.append("summary does not contain all 18 valid cells")
    expected_controls = {
        "strong_proxy_bias_pass": True,
        "strong_latent_gain_cells": 0,
        "revealed_latent_oracle_control_pass": True,
        "condition_degradation_pass": True,
    }
    for key, expected in expected_controls.items():
        if summary.get(key) != expected:
            errors.append(f"summary {key} is {summary.get(key)!r} instead of {expected!r}")

    if not table_path.is_file() or len(table_path.read_text(encoding="utf-8").splitlines()) != 19:
        errors.append("18-cell summary table is missing or incomplete")
    result_card = yaml.safe_load(card_path.read_text(encoding="utf-8"))
    if result_card.get("decision") != "REJECTED_FAILED_PREDICTION_AND_DOMINATED":
        errors.append("result card decision is inconsistent")
    prereg = yaml.safe_load(prereg_path.read_text(encoding="utf-8"))
    if prereg.get("code_commit") != "8b167359c3c114412af397ab69d30875d3fa1bdf" or prereg.get("status") != "COMPLETED_FAIL":
        errors.append("preregistration binding or terminal status is inconsistent")

    state = yaml.safe_load((ROOT / "research_state.yaml").read_text(encoding="utf-8"))
    current_active = set(state.get("branches", {}).get("active", []))
    if not {"H-001", "H-005", "H-014"} <= current_active or "H-021" in current_active:
        errors.append("current portfolio does not preserve the H-021 terminal decision and its three historical survivors")
    if state.get("budget", {}).get("used_units", 0) < 56:
        errors.append("current budget lost the H-021 E0 unit")
    if (ROOT / "05_hypotheses/active/H-021.yaml").exists() or not (ROOT / "05_hypotheses/rejected/H-021.yaml").is_file():
        errors.append("H-021 was not moved from active to rejected")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    if "v0.8.0" not in readme or "-0.004589" not in readme:
        errors.append("README lacks the H-021 versioned result summary")
    decision_log = (ROOT / "09_decisions/decision_log.md").read_text(encoding="utf-8")
    if "D-0013" not in decision_log or "REJECT_H021_NO_INCREMENTAL_GAIN" not in decision_log:
        errors.append("H-021 terminal decision is missing")

    output = {
        "experiment_id": payload.get("experiment_id"),
        "raw_rows": len(rows),
        "valid_cells": len(summary.get("cells", [])),
        "outcome": payload.get("preregistered_outcome"),
        "strong_latent_gain_cells": summary.get("strong_latent_gain_cells"),
        "raw_sha256": actual_hash,
        "errors": errors,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
