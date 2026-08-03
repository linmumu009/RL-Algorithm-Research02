from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_FORMAL_HASHES = {
    "07_results/raw/e0_suite_results.json": "6bcc39b76033f5639a5b37311f5c21e92154622e6495cada399655b40752b688",
    "07_results/raw/e0_h021_results.json": "cba8540ab7570e874e99b9adb50ddc6b85a47ae78e5b0d217d6724014105896b",
    "07_results/raw/e0_h027_results.json": "8bbf6df31fbb17a40e3bdd723dd9edc70343da074ffe78b6f194ede1ef0190ac",
    "07_results/raw/e0_h033_results.json": "f07f69762e7e2b51d288feb1a76aebe9e7adcf4c8bf88fd626f82d70bef5f695",
    "07_results/raw/e0_h039_results.json": "9df09d3e5f5a837b40dbcb5af1b2e89f131806258932899b2a2490cffa5792bf",
}


def canonical_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main() -> None:
    errors: list[str] = []
    manifest_path = ROOT / "10_deliverables/cycle_1_evidence_manifest.csv"
    handoff_path = ROOT / "10_deliverables/cycle_1_final_evidence_handoff.md"
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))
    if len(records) != 57:
        errors.append(f"manifest contains {len(records)} records instead of 57")
    paths = [record["path"] for record in records]
    if len(paths) != len(set(paths)):
        errors.append("manifest paths are not unique")

    hash_failures: list[str] = []
    size_failures: list[str] = []
    for record in records:
        path = ROOT / record["path"]
        if not path.is_file():
            hash_failures.append(record["path"])
            continue
        if path.stat().st_size != int(record["bytes"]):
            size_failures.append(record["path"])
        if canonical_hash(path) != record["sha256_lf_normalized"]:
            hash_failures.append(record["path"])
    if hash_failures:
        errors.append(f"manifest hash failures: {hash_failures}")
    if size_failures:
        errors.append(f"manifest byte-size failures: {size_failures}")

    categories = Counter(record["category"] for record in records)
    expected_counts = {
        "governance": 8,
        "corpus": 7,
        "deliverable": 14,
        "formal_result": 5,
        "result_card": 10,
        "preregistration": 10,
        "paused_survivor": 3,
    }
    if dict(categories) != expected_counts:
        errors.append(f"manifest category counts changed: {dict(categories)}")
    for relative, expected in EXPECTED_FORMAL_HASHES.items():
        if canonical_hash(ROOT / relative) != expected:
            errors.append(f"formal result hash changed: {relative}")

    inventory_rows = list(csv.DictReader((ROOT / "01_corpus/inventory.csv").open(encoding="utf-8-sig", newline="")))
    lineage = json.loads((ROOT / "05_hypotheses/lineage_graph.json").read_text(encoding="utf-8"))
    result_cards = list((ROOT / "07_results/result_cards").glob("R-E0-*.yaml"))
    preregistrations = list((ROOT / "06_experiments/preregistrations").glob("E0-*.yaml"))
    paused_cards = list((ROOT / "05_hypotheses/paused").glob("H-*.yaml"))
    rejected_cards = list((ROOT / "05_hypotheses/rejected").glob("H-*.yaml"))
    active_cards = list((ROOT / "05_hypotheses/active").glob("H-*.yaml"))
    if len(inventory_rows) != 237 or len({row["arxiv_id"] for row in inventory_rows}) != 237:
        errors.append("handoff inventory is not 237 unique papers")
    if len(lineage.get("nodes", [])) != 44:
        errors.append("handoff lineage is not 44 nodes")
    if len(result_cards) != 10 or len(preregistrations) != 10:
        errors.append("handoff does not preserve 10 result cards and preregistrations")
    if len(paused_cards) != 3 or len(rejected_cards) != 41 or active_cards:
        errors.append("handoff branch counts are not 0 active / 3 paused / 41 rejected")

    state = yaml.safe_load((ROOT / "research_state.yaml").read_text(encoding="utf-8"))
    if state.get("phase_status") != "discovery_cycle_1_archived_handoff_ready":
        errors.append("research_state is not handoff-ready")
    if state.get("latest_decision", {}).get("decision_id") != "D-0027":
        errors.append("research_state does not record D-0027")
    if state.get("budget", {}).get("used_units") != 71:
        errors.append("handoff changed the used budget")
    if set(state.get("branches", {}).get("paused", [])) != {"H-001", "H-005", "H-014"}:
        errors.append("handoff paused survivor set changed")

    handoff = handoff_path.read_text(encoding="utf-8")
    required_tokens = [
        "P7_CLOSED_GLOBAL_FALLBACK / G6_NOT_PASSED",
        "正式 E0 分支 | 10",
        "H-001",
        "H-005",
        "H-014",
        "6bcc39b76033f5639a5b37311f5c21e92154622e6495cada399655b40752b688",
        "重启清单",
    ]
    for token in required_tokens:
        if token not in handoff:
            errors.append(f"handoff report lacks {token}")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    if "v0.13.1" not in readme or "cycle_1_final_evidence_handoff.md" not in readme:
        errors.append("README lacks final handoff entry")
    decision_log = (ROOT / "09_decisions/decision_log.md").read_text(encoding="utf-8")
    if "D-0027" not in decision_log or "COMPLETE_CYCLE_1_EVIDENCE_HANDOFF" not in decision_log:
        errors.append("final handoff decision is missing")

    output = {
        "manifest_records": len(records),
        "manifest_categories": dict(categories),
        "inventory_records": len(inventory_rows),
        "lineage_nodes": len(lineage.get("nodes", [])),
        "formal_result_sets": len(EXPECTED_FORMAL_HASHES),
        "result_cards": len(result_cards),
        "preregistrations": len(preregistrations),
        "active_branches": len(active_cards),
        "paused_branches": len(paused_cards),
        "rejected_branches": len(rejected_cards),
        "budget_used": state.get("budget", {}).get("used_units"),
        "errors": errors,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
