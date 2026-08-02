from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_IDS = {"H-001", "H-004", "H-005", "H-008", "H-014", "H-018"}
SURVIVORS = {"H-001", "H-005", "H-014"}
REJECTED_AT_E0 = {"H-004", "H-008", "H-018"}
EXPECTED_OUTCOMES = {
    "H-001": "PASS",
    "H-004": "FAIL",
    "H-005": "PASS",
    "H-008": "PASS",
    "H-014": "PASS",
    "H-018": "FAIL",
}
CODE_COMMIT = "34fea81eb28bdba546580ba91e68d1cca5065805"
RAW_SHA256 = "6bcc39b76033f5639a5b37311f5c21e92154622e6495cada399655b40752b688"
RESULT_KEYS = {"result_id", "experiment_id", "raw_artifacts", "primary_metric", "decision_relevance", "created_at"}


def finite_numbers(value: object) -> bool:
    if isinstance(value, dict):
        return all(finite_numbers(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_numbers(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def main() -> None:
    errors: list[str] = []
    raw_path = ROOT / "07_results/raw/e0_suite_results.json"
    table_path = ROOT / "07_results/tables/e0_summary.csv"
    if not raw_path.is_file():
        errors.append("raw E0 suite result is missing")
        payload = {"experiments": []}
    else:
        raw_bytes = raw_path.read_bytes()
        payload = json.loads(raw_bytes.decode("utf-8"))
        canonical_bytes = raw_bytes.replace(b"\r\n", b"\n")
        if hashlib.sha256(canonical_bytes).hexdigest() != RAW_SHA256:
            errors.append("raw E0 suite checksum differs from the first formal run")

    if payload.get("suite_id") != "Q001-E0-v1":
        errors.append("unexpected E0 suite ID")
    if payload.get("language_model_training") is not False:
        errors.append("E0 artifact does not explicitly prohibit language-model training")
    if payload.get("seeds") != [11, 23, 47, 89, 131]:
        errors.append("E0 seeds differ from the frozen preregistration")
    if not finite_numbers(payload):
        errors.append("E0 raw artifact contains non-finite values")
    checksum_path = ROOT / "07_results/raw/e0_suite_results.sha256"
    if not checksum_path.is_file() or RAW_SHA256 not in checksum_path.read_text(encoding="utf-8"):
        errors.append("raw E0 checksum sidecar is missing or incorrect")

    experiments = {item["hypothesis_id"]: item for item in payload.get("experiments", [])}
    if set(experiments) != EXPECTED_IDS:
        errors.append("E0 raw artifact does not contain exactly the six approved branches")
    for hypothesis_id, expected in EXPECTED_OUTCOMES.items():
        if experiments.get(hypothesis_id, {}).get("preregistered_outcome") != expected:
            errors.append(f"unexpected preregistered outcome for {hypothesis_id}")
    if experiments.get("H-008", {}).get("equivalence_flag") is not True:
        errors.append("H-008 equivalence flag is absent")

    if not table_path.is_file():
        errors.append("E0 summary table is missing")
        table_rows = []
    else:
        with table_path.open(encoding="utf-8", newline="") as stream:
            table_rows = list(csv.DictReader(stream))
    if {row.get("hypothesis_id") for row in table_rows} != EXPECTED_IDS:
        errors.append("E0 summary table branch IDs are incomplete")

    prereg_paths = [
        ROOT / "06_experiments/preregistrations" / f"E0-{hypothesis_id.replace('H-', 'H')}.yaml"
        for hypothesis_id in sorted(EXPECTED_IDS)
    ]
    if len(prereg_paths) != 6:
        errors.append("expected six E0 preregistrations")
    for path in prereg_paths:
        prereg_text = path.read_text(encoding="utf-8")
        match = re.search(r"(?m)^code_commit:\s*(\S+)", prereg_text)
        if match is None or match.group(1) != CODE_COMMIT:
            errors.append(f"{path.name} is not bound to the frozen code commit")

    cards = sorted((ROOT / "07_results/result_cards").glob("R-E0-H*.yaml"))
    if len(cards) != 6:
        errors.append(f"result card count is {len(cards)} instead of 6")
    for path in cards:
        card = yaml.safe_load(path.read_text(encoding="utf-8"))
        missing = RESULT_KEYS - set(card)
        if missing:
            errors.append(f"{path.name} lacks result keys: {sorted(missing)}")
        for artifact in card.get("raw_artifacts", []):
            if not (ROOT / artifact).is_file():
                errors.append(f"{path.name} references missing artifact {artifact}")

    active_ids = {path.stem for path in (ROOT / "05_hypotheses/active").glob("H-*.yaml")}
    if not SURVIVORS <= active_ids:
        errors.append(f"one or more E0 survivor files are absent: {sorted(SURVIVORS - active_ids)}")
    rejected_ids = {path.stem for path in (ROOT / "05_hypotheses/rejected").glob("H-*.yaml")}
    if not REJECTED_AT_E0 <= rejected_ids:
        errors.append("one or more E0 rejections are absent from rejected cards")

    state = yaml.safe_load((ROOT / "research_state.yaml").read_text(encoding="utf-8"))
    state_active = set(state.get("branches", {}).get("active", []))
    if not SURVIVORS <= state_active:
        errors.append("research_state no longer preserves all E0 survivors")
    if REJECTED_AT_E0 & state_active:
        errors.append("an E0-rejected branch was reactivated")
    if state.get("budget", {}).get("used_units", 0) < 51:
        errors.append("research_state budget lost the six E0 units")

    required_reports = [
        "08_reviews/local_reviews/e0_review.md",
        "10_deliverables/e0_experimental_report.md",
    ]
    for relative in required_reports:
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size < 500:
            errors.append(f"missing or short E0 report: {relative}")

    decision_log = (ROOT / "09_decisions/decision_log.md").read_text(encoding="utf-8")
    if "D-0009" not in decision_log or "E0_REVIEW_RETAIN_THREE" not in decision_log:
        errors.append("E0 project-level decision is absent")

    summary = {
        "suite_id": payload.get("suite_id"),
        "experiments": len(experiments),
        "preregistered_passes": sum(value == "PASS" for value in EXPECTED_OUTCOMES.values()),
        "final_survivors": len(SURVIVORS),
        "failed_predictions": 2,
        "equivalent_rejections": 1,
        "result_cards": len(cards),
        "errors": errors,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
