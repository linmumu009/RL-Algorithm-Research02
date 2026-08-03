from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
PAUSED_SURVIVORS = {"H-001", "H-005", "H-014"}


def main() -> None:
    errors: list[str] = []
    state = yaml.safe_load((ROOT / "research_state.yaml").read_text(encoding="utf-8"))
    if state.get("current_phase") != "P7_CLOSED_GLOBAL_FALLBACK":
        errors.append("current phase is not the closed global-fallback phase")
    if state.get("current_gate") != "G6_NOT_PASSED":
        errors.append("G6 terminal status is inconsistent")
    if state.get("branches", {}).get("active") != []:
        errors.append("active portfolio is not empty after cycle closure")
    if set(state.get("branches", {}).get("paused", [])) != PAUSED_SURVIVORS:
        errors.append("paused survivor set is not H-001/H-005/H-014")
    if state.get("budget", {}).get("used_units") != 71:
        errors.append("cycle closure changed the used budget")
    latest_decision = state.get("latest_decision", {}).get("decision_id")
    if latest_decision not in {"D-0026", "D-0027", "D-0028"}:
        errors.append("research_state does not retain the cycle closure, handoff, or terminal assessment decision")
    if len(state.get("blockers", [])) < 2:
        errors.append("G6/reserve closure reasons are missing")

    active_cards = {path.stem for path in (ROOT / "05_hypotheses/active").glob("H-*.yaml")}
    paused_cards = {path.stem for path in (ROOT / "05_hypotheses/paused").glob("H-*.yaml")}
    if active_cards:
        errors.append(f"active hypothesis cards remain: {sorted(active_cards)}")
    if paused_cards != PAUSED_SURVIVORS:
        errors.append("paused hypothesis cards are inconsistent")
    for hypothesis_id in PAUSED_SURVIVORS:
        card = yaml.safe_load((ROOT / f"05_hypotheses/paused/{hypothesis_id}.yaml").read_text(encoding="utf-8"))
        if card.get("status") != "E0_VALIDATED_PAUSED_GLOBAL_FALLBACK":
            errors.append(f"{hypothesis_id} is not marked as E0-validated/paused")

    lineage = json.loads((ROOT / "05_hypotheses/lineage_graph.json").read_text(encoding="utf-8"))
    statuses = {node["id"]: node.get("status") for node in lineage.get("nodes", [])}
    for hypothesis_id in PAUSED_SURVIVORS:
        if statuses.get(hypothesis_id) != "E0_VALIDATED_PAUSED_GLOBAL_FALLBACK":
            errors.append(f"lineage status is inconsistent for {hypothesis_id}")

    report = ROOT / "10_deliverables/p7_global_fallback_governance_review.md"
    if not report.is_file() or report.stat().st_size < 3000:
        errors.append("global fallback governance report is missing or short")
    else:
        report_text = report.read_text(encoding="utf-8")
        for token in ("CLOSE_DISCOVERY_CYCLE_GLOBAL_FALLBACK", "71/100", "剩余预算低于 30%", "H-001", "H-005", "H-014"):
            if token not in report_text:
                errors.append(f"governance report lacks {token}")

    with (ROOT / "09_decisions/budget_ledger.csv").open(encoding="utf-8-sig", newline="") as handle:
        ledger = list(csv.DictReader(handle))
    ledger_by_id = {row.get("entry_id"): row for row in ledger}
    closure_entry = ledger_by_id.get("B-0016", {})
    handoff_entry = ledger_by_id.get("B-0017", {})
    assessment_entry = ledger_by_id.get("B-0018", {})
    if closure_entry.get("units") != "0" or closure_entry.get("cumulative_units") != "71":
        errors.append("zero-cost closure ledger entry is inconsistent")
    if latest_decision in {"D-0027", "D-0028"} and (
        handoff_entry.get("units") != "0"
        or handoff_entry.get("cumulative_units") != "71"
        or handoff_entry.get("decision_id") != "D-0027"
    ):
        errors.append("zero-cost final handoff ledger entry is inconsistent")
    if latest_decision == "D-0028" and (
        assessment_entry.get("units") != "0"
        or assessment_entry.get("cumulative_units") != "71"
        or assessment_entry.get("decision_id") != "D-0028"
    ):
        errors.append("zero-cost terminal assessment ledger entry is inconsistent")
    decision_log = (ROOT / "09_decisions/decision_log.md").read_text(encoding="utf-8")
    if "D-0026" not in decision_log or "CLOSE_DISCOVERY_CYCLE_GLOBAL_FALLBACK" not in decision_log:
        errors.append("cycle closure decision is missing")
    charter = (ROOT / "PROJECT_CHARTER.md").read_text(encoding="utf-8")
    if "第一发现周期" not in charter or "71/100" not in charter:
        errors.append("project charter lacks cycle closure status")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    if "v0.13.0" not in readme or "P7_CLOSED_GLOBAL_FALLBACK" not in readme or "29 单位" not in readme:
        errors.append("README lacks the cycle closure summary")

    output = {
        "phase": state.get("current_phase"),
        "gate": state.get("current_gate"),
        "active_branches": len(active_cards),
        "paused_survivors": len(paused_cards),
        "budget_used": state.get("budget", {}).get("used_units"),
        "budget_remaining": state.get("budget", {}).get("total_units", 0) - state.get("budget", {}).get("used_units", 0),
        "latest_decision": state.get("latest_decision", {}).get("decision_id"),
        "errors": errors,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
