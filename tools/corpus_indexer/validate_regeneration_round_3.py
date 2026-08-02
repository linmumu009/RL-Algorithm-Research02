from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
ROUND_IDS = {"H-033", "H-034", "H-035", "H-036", "H-037", "H-038"}
ACTIVE_IDS = {"H-001", "H-005", "H-014", "H-033"}
ROUND_REJECTED = {"H-034", "H-035", "H-036", "H-037", "H-038"}
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


def has_front_matter_summary(markdown: str) -> bool:
    lower = markdown.lower()
    if "abstract" in lower:
        return True
    introduction = re.search(r"(?mi)^##?\s*(?:1[.\s]+)?introduction\b", markdown)
    prefix = markdown[: introduction.start()] if introduction else ""
    return len(re.sub(r"\s+", " ", prefix).strip()) >= 500


def main() -> None:
    errors: list[str] = []
    inventory = read_csv("01_corpus/inventory.csv")
    supplement = read_csv("02_literature/extended/p7_regeneration_round_3_supplement.csv")
    candidates = read_csv("05_hypotheses/regeneration_round_3.csv")
    inventory_ids = {row["arxiv_id"] for row in inventory}

    if len(inventory) != 231 or len(inventory_ids) != 231:
        errors.append("inventory does not contain exactly 231 unique arXiv papers")
    if any(row["readable"] != "true" for row in inventory):
        errors.append("inventory contains unreadable PDFs")
    if any(not row["abstract"] for row in inventory):
        errors.append("inventory contains missing official abstracts")

    if len(supplement) != 12 or {row["arxiv_id"] for row in supplement} - inventory_ids:
        errors.append("round-3 supplement is incomplete or absent from inventory")
    for row in supplement:
        pdf = ROOT / row["pdf_path"]
        markdown = ROOT / row["markdown_path"]
        if row["markdown_status"] != "parsed":
            errors.append(f"unparsed round-3 paper: {row['arxiv_id']}")
        if not pdf.is_file() or pdf.stat().st_size < 10_000:
            errors.append(f"missing or short PDF: {row['pdf_path']}")
        if not markdown.is_file() or markdown.stat().st_size < 10_000:
            errors.append(f"missing or short Markdown: {row['markdown_path']}")
            continue
        markdown_text = markdown.read_text(encoding="utf-8", errors="replace")
        markdown_lower = markdown_text.lower()
        if not markdown_text.startswith("# "):
            errors.append(f"Markdown lacks title: {row['markdown_path']}")
        if not has_front_matter_summary(markdown_text):
            errors.append(f"Markdown lacks abstract-equivalent front matter: {row['markdown_path']}")
        if "references" not in markdown_lower:
            errors.append(f"Markdown lacks references: {row['markdown_path']}")

    if len(candidates) != 6 or {row["hypothesis_id"] for row in candidates} != ROUND_IDS:
        errors.append("round 3 does not contain exactly H-033 through H-038")
    retained = [row for row in candidates if row["status"] == "PREREGISTERED"]
    rejected = [row for row in candidates if row["status"].startswith("REJECTED_")]
    if [row["hypothesis_id"] for row in retained] != ["H-033"] or len(rejected) != 5:
        errors.append("round-3 retention/rejection split is not 1/5")
    if retained:
        row = retained[0]
        if int(row["total_score"]) < 70 or int(row["falsifiability_score"]) < 12 or int(row["difference_score"]) < 10:
            errors.append("H-033 does not satisfy screening thresholds")

    card_paths = [ROOT / "05_hypotheses/active/H-033.yaml"] + [
        ROOT / f"05_hypotheses/rejected/{hypothesis_id}.yaml" for hypothesis_id in sorted(ROUND_REJECTED)
    ]
    for path in card_paths:
        if not path.is_file():
            errors.append(f"missing round-3 card: {path.name}")
            continue
        missing = HYPOTHESIS_KEYS - top_level_keys(path)
        if missing:
            errors.append(f"{path.name} lacks hypothesis keys: {sorted(missing)}")

    active_ids = {path.stem for path in (ROOT / "05_hypotheses/active").glob("H-*.yaml")}
    if active_ids != ACTIVE_IDS:
        errors.append(f"active portfolio is {sorted(active_ids)} instead of {sorted(ACTIVE_IDS)}")
    lineage = json.loads((ROOT / "05_hypotheses/lineage_graph.json").read_text(encoding="utf-8"))
    nodes = lineage.get("nodes", [])
    node_ids = {node["id"] for node in nodes}
    if len(nodes) != 38 or len(node_ids) != 38 or not ROUND_IDS <= node_ids:
        errors.append("lineage graph does not contain 38 unique hypotheses including round 3")

    prereg = ROOT / "06_experiments/preregistrations/E0-H033.yaml"
    if not prereg.is_file():
        errors.append("H-033 E0 preregistration is missing")
    else:
        missing = EXPERIMENT_KEYS - top_level_keys(prereg)
        if missing:
            errors.append(f"H-033 preregistration lacks keys: {sorted(missing)}")
        prereg_data = yaml.safe_load(prereg.read_text(encoding="utf-8"))
        if prereg_data.get("status") != "PREREGISTERED_NOT_RUN":
            errors.append("H-033 preregistration is not in the pre-implementation lifecycle")
        if prereg_data.get("code_commit") != "TO_BE_SET_BEFORE_EXECUTION":
            errors.append("H-033 preregistration is prematurely bound")
        if prereg_data.get("budget_limit") != 1:
            errors.append("H-033 preregistration budget is not one unit")

    required_reports = [
        "01_corpus/metadata/p7_regeneration_round_3_acquisition_report.md",
        "05_hypotheses/novelty_checks/regeneration_round_3.md",
        "05_hypotheses/equivalence_checks/regeneration_round_3.md",
        "05_hypotheses/theory_risk_regeneration_round_3.md",
        "10_deliverables/replacement_hypothesis_screening_round_3.md",
    ]
    for relative in required_reports:
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size < 500:
            errors.append(f"missing or short round-3 report: {relative}")

    state = yaml.safe_load((ROOT / "research_state.yaml").read_text(encoding="utf-8"))
    if set(state.get("branches", {}).get("active", [])) != ACTIVE_IDS:
        errors.append("research_state active portfolio is inconsistent")
    if state.get("budget", {}).get("used_units") != 65:
        errors.append("research_state budget is not 65")
    if state.get("latest_decision", {}).get("decision_id") != "D-0018":
        errors.append("research_state latest decision is not D-0018")
    if state.get("blockers"):
        errors.append("research_state retains a blocker after restoring four active branches")

    decision_log = (ROOT / "09_decisions/decision_log.md").read_text(encoding="utf-8")
    if "D-0018" not in decision_log or "PASS_REGENERATION_ROUND_3_RETAIN_H033" not in decision_log:
        errors.append("round-3 decision is missing")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    if "v0.11.0" not in readme or "231 篇" not in readme or "H-033" not in readme:
        errors.append("README lacks the round-3 version summary")

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
