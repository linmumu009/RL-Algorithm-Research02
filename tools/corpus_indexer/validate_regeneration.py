from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
ROUND_IDS = {"H-021", "H-022", "H-023", "H-024", "H-025", "H-026"}
BASE_SURVIVORS = {"H-001", "H-005", "H-014"}
ROUND_REJECTED = {"H-021", "H-022", "H-023", "H-024", "H-025", "H-026"}
HYPOTHESIS_KEYS = {
    "hypothesis_id",
    "target_problem",
    "causal_claim",
    "mechanism",
    "falsifiable_predictions",
    "cheapest_falsification_test",
    "failure_threshold",
    "estimated_budget",
    "status",
    "repair_count",
}
EXPERIMENT_KEYS = {
    "experiment_id",
    "hypothesis_id",
    "question",
    "prediction",
    "null_result_interpretation",
    "dataset",
    "model",
    "baseline",
    "primary_metric",
    "seed_policy",
    "budget_limit",
    "stop_condition",
    "success_threshold",
    "failure_threshold",
    "code_commit",
}


def read_csv(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def top_level_keys(path: Path) -> set[str]:
    return set(re.findall(r"(?m)^([A-Za-z_][A-Za-z0-9_]*):", path.read_text(encoding="utf-8")))


def main() -> None:
    errors: list[str] = []
    inventory = read_csv("01_corpus/inventory.csv")
    supplement = read_csv("02_literature/extended/p7_regeneration_supplement.csv")
    candidates = read_csv("05_hypotheses/regeneration_round_1.csv")

    if len(inventory) < 219:
        errors.append(f"inventory count is {len(inventory)} instead of at least 219")
    if len({row["arxiv_id"] for row in inventory}) != len(inventory):
        errors.append("inventory arXiv IDs are not unique")
    if any(row["readable"] != "true" for row in inventory):
        errors.append("inventory contains unreadable PDFs")

    if len(supplement) != 6 or {row["arxiv_id"] for row in supplement} - {row["arxiv_id"] for row in inventory}:
        errors.append("P7 regeneration supplement is incomplete or absent from inventory")
    for row in supplement:
        if row["markdown_status"] != "parsed":
            errors.append(f"unparsed regeneration paper: {row['arxiv_id']}")
        pdf = ROOT / row["pdf_path"]
        markdown = ROOT / row["markdown_path"]
        if not pdf.is_file() or pdf.stat().st_size < 10_000:
            errors.append(f"missing or short PDF: {row['pdf_path']}")
        if not markdown.is_file() or markdown.stat().st_size < 1_000:
            errors.append(f"missing or short Markdown: {row['markdown_path']}")

    if len(candidates) != 6 or {row["hypothesis_id"] for row in candidates} != ROUND_IDS:
        errors.append("regeneration round does not contain exactly H-021 through H-026")
    retained = [row for row in candidates if row["status"] == "PREREGISTERED"]
    rejected = [row for row in candidates if row["status"].startswith("REJECTED_")]
    if [row["hypothesis_id"] for row in retained] != ["H-021"] or len(rejected) != 5:
        errors.append("regeneration retention/rejection split is not 1/5")
    if retained:
        row = retained[0]
        if int(row["total_score"]) < 70 or int(row["falsifiability_score"]) < 12 or int(row["difference_score"]) < 10:
            errors.append("H-021 does not satisfy screening thresholds")

    card_paths = [ROOT / f"05_hypotheses/rejected/{hid}.yaml" for hid in sorted(ROUND_REJECTED)]
    for path in card_paths:
        if not path.is_file():
            errors.append(f"missing regeneration card: {path.name}")
            continue
        missing = HYPOTHESIS_KEYS - top_level_keys(path)
        if missing:
            errors.append(f"{path.name} lacks hypothesis keys: {sorted(missing)}")

    active_ids = {path.stem for path in (ROOT / "05_hypotheses/active").glob("H-*.yaml")}
    active_ids |= {path.stem for path in (ROOT / "05_hypotheses/paused").glob("H-*.yaml")}
    if not BASE_SURVIVORS <= active_ids or "H-021" in active_ids:
        errors.append("current portfolio does not preserve the three first-round E0 survivors and H-021 rejection")

    lineage = json.loads((ROOT / "05_hypotheses/lineage_graph.json").read_text(encoding="utf-8"))
    nodes = lineage.get("nodes", [])
    node_ids = {node["id"] for node in nodes}
    if len(nodes) < 32 or len(node_ids) != len(nodes) or not ROUND_IDS <= node_ids:
        errors.append("lineage graph does not preserve the first regeneration round inside the current unique hypothesis set")

    prereg = ROOT / "06_experiments/preregistrations/E0-H021.yaml"
    if not prereg.is_file():
        errors.append("H-021 E0 preregistration is missing")
    else:
        missing = EXPERIMENT_KEYS - top_level_keys(prereg)
        if missing:
            errors.append(f"H-021 preregistration lacks keys: {sorted(missing)}")
        text = prereg.read_text(encoding="utf-8")
        if "code_commit: 8b167359c3c114412af397ab69d30875d3fa1bdf" not in text or "status: COMPLETED_FAIL" not in text or "budget_limit: 1" not in text:
            errors.append("H-021 preregistration is not bound to the frozen completed-fail record")

    required_reports = [
        "01_corpus/metadata/p7_regeneration_acquisition_report.md",
        "05_hypotheses/novelty_checks/regeneration_round_1.md",
        "05_hypotheses/equivalence_checks/regeneration_round_1.md",
        "05_hypotheses/theory_risk_regeneration_round_1.md",
        "10_deliverables/replacement_hypothesis_screening.md",
    ]
    for relative in required_reports:
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size < 500:
            errors.append(f"missing or short regeneration report: {relative}")

    state = yaml.safe_load((ROOT / "research_state.yaml").read_text(encoding="utf-8"))
    state_current = set(state.get("branches", {}).get("active", [])) | set(state.get("branches", {}).get("paused", []))
    if state_current != active_ids:
        errors.append("research_state current portfolio differs from active/paused hypothesis cards")
    if state.get("budget", {}).get("used_units", 0) < 61:
        errors.append("research_state budget lost completed H-021 or H-027 E0 costs")
    if len(active_ids) < 4 and not state.get("blockers"):
        errors.append("research_state omits the current undersized-portfolio blocker")

    decision_log = (ROOT / "09_decisions/decision_log.md").read_text(encoding="utf-8")
    if "D-0010" not in decision_log or "PASS_TARGETED_REGENERATION_RETAIN_H021" not in decision_log:
        errors.append("targeted regeneration decision is missing")

    summary = {
        "inventory_records": len(inventory),
        "new_papers": len(supplement),
        "replacement_candidates": len(candidates),
        "retained": len(retained),
        "screened_out": len(rejected),
        "historically_retained_for_e0": len(retained),
        "active_portfolio_after_round_2": len(active_ids),
        "lineage_nodes": len(nodes),
        "errors": errors,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
