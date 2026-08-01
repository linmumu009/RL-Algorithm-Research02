from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MECHANISM_KEYS = {
    "mechanism_id",
    "canonical_name",
    "mathematical_form",
    "changed_quantity",
    "intended_effect",
    "target_failure_mode",
    "required_assumptions",
    "known_benefits",
    "known_side_effects",
    "supporting_papers",
    "contradicting_papers",
    "equivalent_forms",
    "incompatible_mechanisms",
    "open_questions",
}


def read_csv(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def yaml_top_level_keys(path: Path) -> set[str]:
    return {
        match.group(1)
        for match in re.finditer(
            r"(?m)^([A-Za-z_][A-Za-z0-9_]*):", path.read_text(encoding="utf-8")
        )
    }


def main() -> None:
    errors: list[str] = []
    inventory = read_csv("01_corpus/inventory.csv")
    supplement = read_csv("02_literature/extended/p3_targeted_supplement.csv")
    claims = read_csv("02_literature/claim_cards/p3_evidence_claims.csv")
    conflicts = read_csv("03_taxonomy/mechanism_conflicts.csv")
    mechanisms = sorted(
        (ROOT / "02_literature" / "mechanism_cards").glob("M-*.yaml")
    )

    inventory_ids = [row["arxiv_id"] for row in inventory]
    inventory_paper_ids = {row["paper_id"] for row in inventory}
    if len(inventory) < 200:
        errors.append(f"inventory contains only {len(inventory)} records")
    if len(inventory_ids) != len(set(inventory_ids)):
        errors.append("inventory arxiv_id values are not unique")
    if any(row["readable"] != "true" for row in inventory):
        errors.append("inventory contains unreadable PDFs")

    if len(supplement) != 20:
        errors.append(f"targeted supplement has {len(supplement)} rows instead of 20")
    if any(row["markdown_status"] != "parsed" for row in supplement):
        errors.append("one or more supplement papers are not marked parsed")
    for row in supplement:
        pdf = ROOT / row["pdf_path"]
        if not pdf.is_file() or pdf.stat().st_size < 10_000:
            errors.append(f"missing or short supplement PDF: {row['pdf_path']}")

    if len(claims) != 20:
        errors.append(f"P3 evidence table has {len(claims)} rows instead of 20")
    claim_ids = [row["claim_id"] for row in claims]
    if len(claim_ids) != len(set(claim_ids)):
        errors.append("P3 claim IDs are not unique")
    for row in claims:
        if row["paper_id"] not in inventory_paper_ids:
            errors.append(f"claim references unknown paper ID: {row['paper_id']}")
        markdown = ROOT / row["markdown_path"]
        if not markdown.is_file() or markdown.stat().st_size < 1_000:
            errors.append(f"missing or short MinerU markdown: {row['markdown_path']}")

    if len(mechanisms) != 15:
        errors.append(f"mechanism card count is {len(mechanisms)} instead of 15")
    mechanism_ids: list[str] = []
    referenced_papers: set[str] = set()
    for card in mechanisms:
        text = card.read_text(encoding="utf-8")
        referenced_papers.update(re.findall(r"P-ARXIV-\d+", text))
        keys = yaml_top_level_keys(card)
        missing = MECHANISM_KEYS - keys
        if missing:
            errors.append(f"{card.name} lacks keys: {sorted(missing)}")
        match = re.search(r"(?m)^mechanism_id:\s*(\S+)", text)
        if match:
            mechanism_ids.append(match.group(1))
    if len(mechanism_ids) != len(set(mechanism_ids)):
        errors.append("mechanism IDs are not unique")
    unknown_papers = referenced_papers - inventory_paper_ids
    if unknown_papers:
        errors.append(f"mechanism cards reference unknown papers: {sorted(unknown_papers)}")

    if len(conflicts) < 10:
        errors.append(f"mechanism conflict table has only {len(conflicts)} rows")

    knowledge_map = (ROOT / "10_deliverables/knowledge_map.md").read_text(
        encoding="utf-8"
    )
    required_g3_phrases = [
        "为什么现有算法在某类场景失败",
        "哪些结论只有相关性证据",
        "哪些机制本质等价",
        "哪些“新组合”已有工作覆盖",
        "哪些问题能用低成本实验辨别不同解释",
    ]
    for phrase in required_g3_phrases:
        if phrase not in knowledge_map:
            errors.append(f"knowledge map does not answer G3 phrase: {phrase}")

    low_cost = (ROOT / "04_problems/baseline_diagnostics/p3_low_cost_discriminators.md").read_text(
        encoding="utf-8"
    )
    test_count = len(re.findall(r"(?m)^\| T\d{2} ", low_cost))
    if test_count < 8:
        errors.append(f"only {test_count} low-cost discriminators found")

    state = (ROOT / "research_state.yaml").read_text(encoding="utf-8")
    if "current_phase: P4_PROBLEM_SELECTION" not in state or "current_gate: G4" not in state:
        errors.append("research state is not paused at P4/G4")

    summary = {
        "inventory_records": len(inventory),
        "targeted_supplement_records": len(supplement),
        "p3_claims": len(claims),
        "mechanism_cards": len(mechanisms),
        "mechanism_conflicts": len(conflicts),
        "low_cost_discriminators": test_count,
        "errors": errors,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
