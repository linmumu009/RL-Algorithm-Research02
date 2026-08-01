from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "01_corpus" / "raw"
INVENTORY_PATH = ROOT / "01_corpus" / "inventory.csv"
DEDUP_PATH = ROOT / "01_corpus" / "dedup_report.md"
METADATA_PATH = ROOT / "01_corpus" / "metadata" / "arxiv_metadata.json"
MANIFEST_PATH = ROOT / "02_literature" / "core15" / "core15_manifest.csv"
ARXIV_RE = re.compile(r"(?P<id>\d{4}\.\d{4,5})(?:v\d+)?")
ATOM = {"a": "http://www.w3.org/2005/Atom"}


def normalize_space(value: str) -> str:
    return " ".join((value or "").split())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_arxiv_id(path: Path) -> str:
    match = ARXIV_RE.search(path.name)
    return match.group("id") if match else ""


def load_manifest() -> dict[str, dict[str, str]]:
    with MANIFEST_PATH.open(encoding="utf-8-sig", newline="") as stream:
        return {row["arxiv_id"]: row for row in csv.DictReader(stream)}


def parse_entries(payload: bytes) -> dict[str, dict[str, object]]:
    root = ET.fromstring(payload)
    parsed: dict[str, dict[str, object]] = {}
    for entry in root.findall("a:entry", ATOM):
        raw_id = entry.findtext("a:id", default="", namespaces=ATOM)
        match = ARXIV_RE.search(raw_id)
        if not match:
            continue
        arxiv_id = match.group("id")
        parsed[arxiv_id] = {
            "title": normalize_space(entry.findtext("a:title", default="", namespaces=ATOM)),
            "authors": [
                normalize_space(author.findtext("a:name", default="", namespaces=ATOM))
                for author in entry.findall("a:author", ATOM)
            ],
            "abstract": normalize_space(entry.findtext("a:summary", default="", namespaces=ATOM)),
            "published": entry.findtext("a:published", default="", namespaces=ATOM),
            "updated": entry.findtext("a:updated", default="", namespaces=ATOM),
        }
    return parsed


def fetch_arxiv(ids: list[str], cached: dict[str, object]) -> dict[str, object]:
    missing = [arxiv_id for arxiv_id in ids if arxiv_id not in cached]
    for offset in range(0, len(missing), 30):
        batch = missing[offset : offset + 30]
        query = urllib.parse.urlencode({"id_list": ",".join(batch), "max_results": len(batch)})
        url = f"https://export.arxiv.org/api/query?{query}"
        request = urllib.request.Request(url, headers={"User-Agent": "RL-Algorithm-Research02/0.3 inventory audit"})
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    cached.update(parse_entries(response.read()))
                last_error = None
                break
            except Exception as exc:  # network and transient server errors
                last_error = exc
                time.sleep(2 * (attempt + 1))
        if last_error:
            print(f"metadata warning for batch {batch[0]}..{batch[-1]}: {last_error}")
        time.sleep(1)
    return cached


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch-metadata", action="store_true")
    args = parser.parse_args()

    pdf_paths = sorted(RAW_DIR.rglob("*.pdf"))
    manifest = load_manifest()
    metadata: dict[str, object] = {}
    if METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    arxiv_ids = sorted({extract_arxiv_id(path) for path in pdf_paths if extract_arxiv_id(path)})
    if args.fetch_metadata:
        metadata = fetch_arxiv(arxiv_ids, metadata)
        METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    rows: list[dict[str, object]] = []
    for path in pdf_paths:
        relative = path.relative_to(ROOT).as_posix()
        arxiv_id = extract_arxiv_id(path)
        record = metadata.get(arxiv_id, {}) if arxiv_id else {}
        readable = True
        page_count = 0
        pdf_title = ""
        first_page_chars = 0
        error = ""
        try:
            reader = PdfReader(str(path), strict=False)
            page_count = len(reader.pages)
            pdf_title = normalize_space(str((reader.metadata or {}).get("/Title") or ""))
            first_page_chars = len((reader.pages[0].extract_text() or "").strip()) if reader.pages else 0
        except Exception as exc:
            readable = False
            error = f"{type(exc).__name__}: {exc}"

        title = str(record.get("title") or pdf_title or path.stem)
        published = str(record.get("published") or "")
        core = manifest.get(arxiv_id)
        rows.append(
            {
                "paper_id": f"P-ARXIV-{arxiv_id.replace('.', '')}" if arxiv_id else f"P-FILE-{sha256(path)[:12]}",
                "arxiv_id": arxiv_id,
                "title": title,
                "authors": "; ".join(record.get("authors", [])),
                "year": published[:4] if published else (arxiv_id[:2] if arxiv_id else ""),
                "abstract": str(record.get("abstract") or ""),
                "source_file": relative,
                "source_url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "",
                "corpus_tier": "core15" if core else ("core_candidate" if "core_candidates" in path.parts else "extended"),
                "mineru_markdown": core["markdown_path"] if core else "",
                "file_hash_sha256": sha256(path),
                "file_size_bytes": path.stat().st_size,
                "page_count": page_count,
                "readable": str(readable).lower(),
                "first_page_text_chars": first_page_chars,
                "main_version": "true",
                "duplicate_of": "",
                "exclusion_status": "",
                "error": error,
            }
        )

    by_hash: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    by_arxiv: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_hash[str(row["file_hash_sha256"])].append(row)
        if row["arxiv_id"]:
            by_arxiv[str(row["arxiv_id"])].append(row)

    duplicate_groups: list[tuple[str, list[dict[str, object]]]] = []
    for key, group in by_arxiv.items():
        if len(group) > 1:
            ordered = sorted(group, key=lambda item: (item["corpus_tier"] != "core15", str(item["source_file"])))
            main_id = str(ordered[0]["paper_id"])
            for duplicate in ordered[1:]:
                duplicate["main_version"] = "false"
                duplicate["duplicate_of"] = main_id
            duplicate_groups.append((f"arxiv:{key}", ordered))

    exact_duplicate_groups = [(key, group) for key, group in by_hash.items() if len(group) > 1]

    fieldnames = list(rows[0].keys()) if rows else []
    with INVENTORY_PATH.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    missing_metadata = [row for row in rows if not row["abstract"]]
    unreadable = [row for row in rows if row["readable"] != "true"]
    report = [
        "# Corpus Deduplication and Integrity Report",
        "",
        f"- PDF files scanned: {len(rows)}",
        f"- Unique arXiv IDs: {len(by_arxiv)}",
        f"- Exact hash duplicate groups: {len(exact_duplicate_groups)}",
        f"- Repeated arXiv ID groups: {len(duplicate_groups)}",
        f"- Unreadable PDFs: {len(unreadable)}",
        f"- Records without API abstract metadata: {len(missing_metadata)}",
        "",
        "## Duplicate groups",
        "",
    ]
    if not duplicate_groups and not exact_duplicate_groups:
        report.append("No duplicate groups were detected by arXiv ID or SHA-256 hash.")
    else:
        for label, group in duplicate_groups:
            report.append(f"- {label}: " + ", ".join(str(item["source_file"]) for item in group))
        for key, group in exact_duplicate_groups:
            report.append(f"- sha256:{key[:12]}: " + ", ".join(str(item["source_file"]) for item in group))
    report.extend(["", "## Integrity exceptions", ""])
    if unreadable:
        for row in unreadable:
            report.append(f"- `{row['source_file']}`: {row['error']}")
    else:
        report.append("All PDFs are readable and contain at least one page.")
    report.extend(["", "## Metadata gaps", ""])
    if missing_metadata:
        for row in missing_metadata:
            report.append(f"- `{row['paper_id']}` `{row['source_file']}`")
    else:
        report.append("All arXiv-backed records have title, author, year, and abstract metadata.")
    DEDUP_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(json.dumps({
        "pdfs": len(rows),
        "unique_arxiv_ids": len(by_arxiv),
        "metadata_records": len(metadata),
        "missing_metadata": len(missing_metadata),
        "unreadable": len(unreadable),
        "duplicate_arxiv_groups": len(duplicate_groups),
        "exact_duplicate_groups": len(exact_duplicate_groups),
    }, indent=2))


if __name__ == "__main__":
    main()
