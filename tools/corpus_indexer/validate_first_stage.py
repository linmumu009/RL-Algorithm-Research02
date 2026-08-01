from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REQUIRED_CARD_KEYS = {
    "paper_id",
    "title",
    "source_file",
    "parsed_text",
    "problem_addressed",
    "core_claim",
    "mechanism_changes",
    "main_evidence",
    "reported_failures",
    "limitations",
    "confidence",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    errors: list[str] = []
    inventory = read_csv(ROOT / "01_corpus" / "inventory.csv")
    manifest = read_csv(ROOT / "02_literature" / "core15" / "core15_manifest.csv")
    matrix = read_csv(ROOT / "03_taxonomy" / "mechanism_matrix.csv")

    inventory_ids = [row["paper_id"] for row in inventory]
    if len(inventory) < 180:
        errors.append(f"inventory contains only {len(inventory)} records")
    if len(inventory_ids) != len(set(inventory_ids)):
        errors.append("inventory paper_id values are not unique")
    if any(row["readable"] != "true" for row in inventory):
        errors.append("inventory contains unreadable PDFs")
    for row in inventory:
        if not (ROOT / row["source_file"]).is_file():
            errors.append(f"missing inventory source: {row['source_file']}")

    if len(manifest) != 15:
        errors.append(f"core manifest contains {len(manifest)} rows instead of 15")
    manifest_ids = {row["paper_id"] for row in manifest}
    if {row["paper_id"] for row in matrix} != manifest_ids:
        errors.append("mechanism matrix and core manifest paper IDs differ")

    for row in manifest:
        pdf = ROOT / row["pdf_path"]
        markdown = ROOT / row["markdown_path"]
        card = ROOT / "02_literature" / "paper_cards" / f"{row['paper_id']}.yaml"
        if not pdf.is_file():
            errors.append(f"missing core PDF: {pdf.relative_to(ROOT)}")
        if not markdown.is_file() or markdown.stat().st_size < 1000:
            errors.append(f"missing or short MinerU markdown: {markdown.relative_to(ROOT)}")
        if not card.is_file():
            errors.append(f"missing Paper Card: {card.relative_to(ROOT)}")
            continue
        keys = {
            match.group(1)
            for match in re.finditer(r"(?m)^([A-Za-z_][A-Za-z0-9_]*):", card.read_text(encoding="utf-8"))
        }
        missing_keys = REQUIRED_CARD_KEYS - keys
        if missing_keys:
            errors.append(f"Paper Card {card.name} lacks {sorted(missing_keys)}")

    for schema in (ROOT / "schemas").glob("*.schema.json"):
        try:
            json.loads(schema.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"invalid JSON schema {schema.name}: {exc}")

    required_outputs = [
        "PROJECT_CHARTER.md",
        "research_state.yaml",
        "01_corpus/inventory.csv",
        "01_corpus/dedup_report.md",
        "03_taxonomy/mechanism_matrix.csv",
        "03_taxonomy/coverage_report.md",
        "10_deliverables/core15_audit.md",
        "02_literature/selection_audit/replacement_recommendations.md",
    ]
    for relative in required_outputs:
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing required output: {relative}")

    summary = {
        "inventory_records": len(inventory),
        "core_manifest_records": len(manifest),
        "paper_cards": len(list((ROOT / "02_literature" / "paper_cards").glob("*.yaml"))),
        "matrix_rows": len(matrix),
        "errors": errors,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
