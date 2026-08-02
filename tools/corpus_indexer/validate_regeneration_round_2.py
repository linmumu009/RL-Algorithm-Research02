from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
ROUND_IDS = {"H-027", "H-028", "H-029", "H-030", "H-031", "H-032"}
ACTIVE_IDS = {"H-001", "H-005", "H-014", "H-027"}
POST_H027_ACTIVE_IDS = {"H-001", "H-005", "H-014"}
ROUND_REJECTED = {"H-028", "H-029", "H-030", "H-031", "H-032"}
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
    supplement = read_csv("02_literature/extended/p7_regeneration_round_2_supplement.csv")
    candidates = read_csv("05_hypotheses/regeneration_round_2.csv")
    inventory_ids = {row["arxiv_id"] for row in inventory}

    if len(inventory) != 219 or len(inventory_ids) != 219:
        errors.append("inventory does not contain exactly 219 unique arXiv papers")
    if any(row["readable"] != "true" for row in inventory):
        errors.append("inventory contains unreadable PDFs")
    if any(not row["abstract"] for row in inventory):
        errors.append("inventory contains missing official abstracts")

    if len(supplement) != 7 or {row["arxiv_id"] for row in supplement} - inventory_ids:
        errors.append("round-2 supplement is incomplete or absent from inventory")
    for row in supplement:
        pdf = ROOT / row["pdf_path"]
        markdown = ROOT / row["markdown_path"]
        if row["markdown_status"] != "parsed":
            errors.append(f"unparsed round-2 paper: {row['arxiv_id']}")
        if not pdf.is_file() or pdf.stat().st_size < 10_000:
            errors.append(f"missing or short PDF: {row['pdf_path']}")
        if not markdown.is_file() or markdown.stat().st_size < 10_000:
            errors.append(f"missing or short Markdown: {row['markdown_path']}")
        else:
            markdown_text = markdown.read_text(encoding="utf-8", errors="replace")
            markdown_lower = markdown_text.lower()
            if not markdown_text.startswith("# ") or "abstract" not in markdown_lower or "references" not in markdown_lower:
                errors.append(f"Markdown lacks title abstract or references: {row['markdown_path']}")

    if len(candidates) != 6 or {row["hypothesis_id"] for row in candidates} != ROUND_IDS:
        errors.append("round 2 does not contain exactly H-027 through H-032")
    retained = [row for row in candidates if row["status"] == "PREREGISTERED"]
    rejected = [row for row in candidates if row["status"].startswith("REJECTED_")]
    if [row["hypothesis_id"] for row in retained] != ["H-027"] or len(rejected) != 5:
        errors.append("round-2 retention/rejection split is not 1/5")
    if retained:
        row = retained[0]
        if int(row["total_score"]) < 70 or int(row["falsifiability_score"]) < 12 or int(row["difference_score"]) < 10:
            errors.append("H-027 does not satisfy screening thresholds")

    h027_active = ROOT / "05_hypotheses/active/H-027.yaml"
    h027_rejected = ROOT / "05_hypotheses/rejected/H-027.yaml"
    h027_card = h027_active if h027_active.is_file() else h027_rejected
    card_paths = [h027_card] + [
        ROOT / f"05_hypotheses/rejected/{hypothesis_id}.yaml" for hypothesis_id in sorted(ROUND_REJECTED)
    ]
    for path in card_paths:
        if not path.is_file():
            errors.append(f"missing round-2 card: {path.name}")
            continue
        missing = HYPOTHESIS_KEYS - top_level_keys(path)
        if missing:
            errors.append(f"{path.name} lacks hypothesis keys: {sorted(missing)}")

    active_ids = {path.stem for path in (ROOT / "05_hypotheses/active").glob("H-*.yaml")}
    expected_active = POST_H027_ACTIVE_IDS if h027_rejected.is_file() else ACTIVE_IDS
    if active_ids != expected_active:
        errors.append(f"active portfolio is {sorted(active_ids)} instead of {sorted(expected_active)}")
    lineage = json.loads((ROOT / "05_hypotheses/lineage_graph.json").read_text(encoding="utf-8"))
    nodes = lineage.get("nodes", [])
    node_ids = {node["id"] for node in nodes}
    if len(nodes) != 32 or len(node_ids) != 32 or not ROUND_IDS <= node_ids:
        errors.append("lineage graph does not contain 32 unique hypotheses including round 2")

    prereg = ROOT / "06_experiments/preregistrations/E0-H027.yaml"
    if not prereg.is_file():
        errors.append("H-027 E0 preregistration is missing")
    else:
        missing = EXPERIMENT_KEYS - top_level_keys(prereg)
        if missing:
            errors.append(f"H-027 preregistration lacks keys: {sorted(missing)}")
        prereg_data = yaml.safe_load(prereg.read_text(encoding="utf-8"))
        lifecycle = prereg_data.get("status")
        code_commit = str(prereg_data.get("code_commit", ""))
        allowed_lifecycle = {
            "PREREGISTERED_NOT_RUN",
            "IMPLEMENTATION_FROZEN_NOT_RUN",
            "BOUND_NOT_RUN",
            "COMPLETED_PASS",
            "COMPLETED_FAIL",
        }
        if lifecycle not in allowed_lifecycle or prereg_data.get("budget_limit") != 1:
            errors.append("H-027 preregistration lifecycle or budget is invalid")
        if lifecycle in {"PREREGISTERED_NOT_RUN", "IMPLEMENTATION_FROZEN_NOT_RUN"}:
            if code_commit != "TO_BE_SET_BEFORE_EXECUTION":
                errors.append("unbound H-027 preregistration already contains a code commit")
        elif not re.fullmatch(r"[0-9a-f]{40}", code_commit):
            errors.append("bound or completed H-027 preregistration lacks an immutable commit")

    required_reports = [
        "01_corpus/metadata/p7_regeneration_round_2_acquisition_report.md",
        "05_hypotheses/novelty_checks/regeneration_round_2.md",
        "05_hypotheses/equivalence_checks/regeneration_round_2.md",
        "05_hypotheses/theory_risk_regeneration_round_2.md",
        "10_deliverables/replacement_hypothesis_screening_round_2.md",
    ]
    for relative in required_reports:
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size < 500:
            errors.append(f"missing or short round-2 report: {relative}")

    state = yaml.safe_load((ROOT / "research_state.yaml").read_text(encoding="utf-8"))
    if set(state.get("branches", {}).get("active", [])) != expected_active:
        errors.append("research_state active portfolio is inconsistent with the H-027 lifecycle")
    expected_budget = 61 if h027_rejected.is_file() else 60
    if state.get("budget", {}).get("used_units") != expected_budget:
        errors.append(f"research_state budget is not {expected_budget}")
    if not h027_rejected.is_file() and state.get("blockers"):
        errors.append("research_state reports a portfolio blocker before H-027 completion")
    if state.get("latest_decision", {}).get("decision_id") not in {"D-0014", "D-0015", "D-0016", "D-0017"}:
        errors.append("research_state latest decision is outside the H-027 lifecycle")

    decision_log = (ROOT / "09_decisions/decision_log.md").read_text(encoding="utf-8")
    if "D-0014" not in decision_log or "PASS_REGENERATION_ROUND_2_RETAIN_H027" not in decision_log:
        errors.append("round-2 decision is missing")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    if "v0.9.0" not in readme or "219 篇" not in readme or "H-027" not in readme:
        errors.append("README lacks the round-2 version summary")

    summary = {
        "inventory_records": len(inventory),
        "new_papers": len(supplement),
        "replacement_candidates": len(candidates),
        "retained": len(retained),
        "screened_out": len(rejected),
        "active_portfolio": len(active_ids),
        "lineage_nodes": len(nodes),
        "errors": errors,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
