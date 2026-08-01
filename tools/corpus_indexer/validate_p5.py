from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
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
ACTIVE_IDS = {"H-001", "H-004", "H-005", "H-008", "H-014", "H-018"}


def read_csv(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def yaml_keys(path: Path) -> set[str]:
    return {
        match.group(1)
        for match in re.finditer(
            r"(?m)^([A-Za-z_][A-Za-z0-9_]*):", path.read_text(encoding="utf-8")
        )
    }


def main() -> None:
    errors: list[str] = []
    inventory = read_csv("01_corpus/inventory.csv")
    initial = read_csv("05_hypotheses/initial_hypotheses.csv")
    supplement = read_csv("02_literature/extended/p5_novelty_supplement.csv")
    active_cards = sorted((ROOT / "05_hypotheses/active").glob("H-*.yaml"))
    rejected_cards = sorted((ROOT / "05_hypotheses/rejected").glob("H-*.yaml"))
    preregistrations = sorted(
        (ROOT / "06_experiments/preregistrations").glob("E0-H*.yaml")
    )

    if len(inventory) < 206:
        errors.append(f"inventory contains only {len(inventory)} records")
    if len({row["arxiv_id"] for row in inventory}) != len(inventory):
        errors.append("inventory arxiv IDs are not unique")
    if any(row["readable"] != "true" for row in inventory):
        errors.append("inventory contains unreadable PDFs")

    if len(supplement) != 6:
        errors.append(f"novelty supplement has {len(supplement)} rows instead of 6")
    if any(row["markdown_status"] != "parsed" for row in supplement):
        errors.append("one or more novelty papers are not marked parsed")
    inventory_arxiv = {row["arxiv_id"] for row in inventory}
    for row in supplement:
        if row["arxiv_id"] not in inventory_arxiv:
            errors.append(f"novelty paper absent from inventory: {row['arxiv_id']}")
        pdf = ROOT / row["pdf_path"]
        if not pdf.is_file() or pdf.stat().st_size < 10_000:
            errors.append(f"missing or short novelty PDF: {row['pdf_path']}")
        markdown = ROOT / row["markdown_path"]
        if not markdown.is_file() or markdown.stat().st_size < 1_000:
            errors.append(f"missing or short novelty Markdown: {row['markdown_path']}")

    if len(initial) != 20:
        errors.append(f"initial hypothesis count is {len(initial)} instead of 20")
    initial_ids = {row["hypothesis_id"] for row in initial}
    if len(initial_ids) != 20:
        errors.append("hypothesis IDs are not unique")
    prereg_rows = [row for row in initial if row["status"] == "PREREGISTERED"]
    rejected_rows = [row for row in initial if row["status"].startswith("REJECTED_")]
    if {row["hypothesis_id"] for row in prereg_rows} != ACTIVE_IDS:
        errors.append("preregistered hypothesis IDs differ from the approved six")
    if len(rejected_rows) != 14:
        errors.append(f"rejected hypothesis count is {len(rejected_rows)} instead of 14")
    families = {row["mechanism_family"] for row in prereg_rows}
    if len(families) < 3:
        errors.append(f"only {len(families)} active mechanism families")
    for row in prereg_rows:
        if int(row["total_score"]) < 70:
            errors.append(f"active score below 70: {row['hypothesis_id']}")
        if int(row["falsifiability_score"]) < 12:
            errors.append(f"active falsifiability below 12: {row['hypothesis_id']}")
        if int(row["difference_score"]) < 10:
            errors.append(f"active difference below 10: {row['hypothesis_id']}")

    if len(active_cards) != 6 or len(rejected_cards) != 14:
        errors.append(
            f"card split is active={len(active_cards)}, rejected={len(rejected_cards)}"
        )
    for card in active_cards + rejected_cards:
        missing = HYPOTHESIS_KEYS - yaml_keys(card)
        if missing:
            errors.append(f"{card.name} lacks hypothesis keys: {sorted(missing)}")

    lineage = json.loads(
        (ROOT / "05_hypotheses/lineage_graph.json").read_text(encoding="utf-8-sig")
    )
    nodes = lineage.get("nodes", [])
    if len(nodes) != 20 or {node["id"] for node in nodes} != initial_ids:
        errors.append("lineage graph does not contain the same 20 hypotheses")
    if {node["id"] for node in nodes if node["status"] == "PREREGISTERED"} != ACTIVE_IDS:
        errors.append("lineage active statuses differ from approved six")

    if len(preregistrations) != 6:
        errors.append(f"E0 preregistration count is {len(preregistrations)} instead of 6")
    prereg_ids: set[str] = set()
    for prereg in preregistrations:
        text = prereg.read_text(encoding="utf-8")
        missing = EXPERIMENT_KEYS - yaml_keys(prereg)
        if missing:
            errors.append(f"{prereg.name} lacks experiment keys: {sorted(missing)}")
        match = re.search(r"(?m)^hypothesis_id:\s*(H-\d+)", text)
        if match:
            prereg_ids.add(match.group(1))
    if prereg_ids != ACTIVE_IDS:
        errors.append("E0 preregistrations do not cover the approved six")

    required_reports = [
        "05_hypotheses/novelty_checks/novelty_report.md",
        "05_hypotheses/equivalence_checks/equivalence_report.md",
        "05_hypotheses/theory_risk_report.md",
        "10_deliverables/hypothesis_screening.md",
    ]
    for relative in required_reports:
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size < 500:
            errors.append(f"missing or short screening report: {relative}")

    decision_log = (ROOT / "09_decisions/decision_log.md").read_text(encoding="utf-8")
    if "D-0006" not in decision_log or "PASS_G5_RETAIN_SIX" not in decision_log:
        errors.append("G5 pass decision is absent from the decision log")

    summary = {
        "inventory_records": len(inventory),
        "initial_hypotheses": len(initial),
        "active_hypotheses": len(active_cards),
        "rejected_hypotheses": len(rejected_cards),
        "active_mechanism_families": len(families),
        "e0_preregistrations": len(preregistrations),
        "errors": errors,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
