from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_CANDIDATES = {"C2-Q01", "C2-Q02", "C2-Q03", "C2-Q04", "C2-Q05"}


def main() -> None:
    errors: list[str] = []
    state = yaml.safe_load((ROOT / "research_state.yaml").read_text(encoding="utf-8"))
    restart = state.get("cycle_2_restart", {})

    if state.get("current_phase") != "P7_CLOSED_GLOBAL_FALLBACK":
        errors.append("cycle-1 closed phase was changed")
    if state.get("current_gate") != "G6_NOT_PASSED":
        errors.append("cycle-1 G6 failure was changed")
    if state.get("phase_status") != "cycle_2_g0_g4_decision_packet_ready_human_review":
        errors.append("cycle-2 decision packet is not at human review")
    if state.get("latest_decision", {}).get("decision_id") != "D-0029":
        errors.append("latest decision is not D-0029")
    if state.get("budget", {}).get("used_units") != 71:
        errors.append("decision preparation changed the used budget")
    if state.get("branches", {}).get("active"):
        errors.append("algorithm branches became active before G4")

    if restart.get("status") != "G0_G4_DECISION_PACKET_READY_HUMAN_REVIEW_REQUIRED":
        errors.append("cycle-2 restart status is inconsistent")
    if restart.get("selected_problem") is not None:
        errors.append("a cycle-2 problem was selected without human approval")
    if restart.get("research_authorized") is not False:
        errors.append("research was authorized before G0/G4 approval")
    if restart.get("budget_requested") is not False or restart.get("budget_approved") is not False:
        errors.append("cycle-2 budget status is not unrequested/unapproved")
    if restart.get("first_cycle_reserve_reallocated") is not False:
        errors.append("cycle-1 reserve was reallocated")
    if set(restart.get("candidate_problem_ids", [])) != EXPECTED_CANDIDATES:
        errors.append("cycle-2 candidate problem set changed")
    if restart.get("candidate_problem_gate_status") != "HOLD_EVIDENCE_REQUIRED":
        errors.append("candidate problems are not held for evidence")

    candidate_path = ROOT / "04_problems/cycle_2_candidate_problem_statements.md"
    packet_path = ROOT / "10_deliverables/cycle_2_g0_g4_decision_packet.md"
    restart_path = ROOT / "09_decisions/restart_records/cycle_2_g0_g4_preparation.md"
    for path in (candidate_path, packet_path, restart_path):
        if not path.is_file() or path.stat().st_size < 1000:
            errors.append(f"cycle-2 decision artifact is missing or short: {path.relative_to(ROOT)}")

    candidates = candidate_path.read_text(encoding="utf-8")
    for token in (*sorted(EXPECTED_CANDIDATES), "HOLD_EVIDENCE_REQUIRED", "可复现性统一记为 0", "明确排除：继续 Q-001", "G4 前必须补齐的事实"):
        if token not in candidates:
            errors.append(f"candidate problem file lacks {token}")

    packet = packet_path.read_text(encoding="utf-8")
    packet_tokens = (
        "G0_G4_DECISION_PACKET_READY / HUMAN_REVIEW_REQUIRED",
        "当前选定问题：`null`",
        "第二周期预算：未申请、未批准",
        "NOT_PASSED_EVIDENCE_AND_HUMAN_DECISION_REQUIRED",
        "APPROVE_G0_AND_SUBMIT_G4_EVIDENCE",
        "DEFER_G4_NO_OBSERVED_PROBLEM",
        "REJECT_CYCLE_2_RESTART",
        "不存在“直接选择一个方向并开始实验”的选项",
    )
    for token in packet_tokens:
        if token not in packet:
            errors.append(f"G0/G4 packet lacks {token}")

    restart_text = restart_path.read_text(encoding="utf-8")
    for token in ("PREPARED_NOT_APPROVED", "研究执行授权：`false`", "选定问题：`null`", "不挪用第一周期 29 单位储备"):
        if token not in restart_text:
            errors.append(f"restart record lacks {token}")

    with (ROOT / "09_decisions/budget_ledger.csv").open(encoding="utf-8-sig", newline="") as handle:
        ledger = list(csv.DictReader(handle))
    entry = next((row for row in ledger if row.get("entry_id") == "B-0019"), None)
    if not entry or entry.get("units") != "0" or entry.get("cumulative_units") != "71" or entry.get("decision_id") != "D-0029":
        errors.append("B-0019 zero-cost decision packet ledger entry is inconsistent")

    decision_log = (ROOT / "09_decisions/decision_log.md").read_text(encoding="utf-8")
    if "D-0029" not in decision_log or "PREPARE_CYCLE2_G0_G4_DECISION_PACKET" not in decision_log:
        errors.append("D-0029 decision is missing")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    if "v0.13.3" not in readme or "HOLD_EVIDENCE_REQUIRED" not in readme:
        errors.append("README lacks cycle-2 preparation version record")

    cycle2_experiment_files = [
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "06_experiments").rglob("*")
        if path.is_file() and ("cycle_2" in path.name.lower() or "c2-" in path.name.lower())
    ]
    if cycle2_experiment_files:
        errors.append(f"cycle-2 experiment artifacts exist before approval: {cycle2_experiment_files}")

    output = {
        "status": restart.get("status"),
        "selected_problem": restart.get("selected_problem"),
        "research_authorized": restart.get("research_authorized"),
        "candidate_problems": len(restart.get("candidate_problem_ids", [])),
        "budget_used": state.get("budget", {}).get("used_units"),
        "cycle2_experiment_files": len(cycle2_experiment_files),
        "errors": errors,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
