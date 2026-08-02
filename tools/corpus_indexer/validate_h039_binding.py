from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_COMMIT = "ddb66391e42bbaf5e63c85949df6c4fac8d32414"
FROZEN_PATHS = [
    "06_experiments/code/h039_sign_certificate_e0.py",
    "06_experiments/configs/e0_h039.yaml",
    "06_experiments/unit_tests/test_h039_sign_certificate_e0.py",
]


def git_output(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def main() -> None:
    errors: list[str] = []
    prereg_path = ROOT / "06_experiments/preregistrations/E0-H039.yaml"
    prereg = yaml.safe_load(prereg_path.read_text(encoding="utf-8"))
    if prereg.get("code_commit") != EXPECTED_COMMIT or prereg.get("status") != "BOUND_NOT_RUN":
        errors.append("H-039 preregistration is not bound to the frozen commit")

    try:
        commit_type = git_output("cat-file", "-t", EXPECTED_COMMIT)
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit_type = ""
    if commit_type != "commit":
        errors.append("frozen H-039 commit is unavailable")

    blob_matches: dict[str, bool] = {}
    for relative in FROZEN_PATHS:
        path = ROOT / relative
        try:
            frozen_blob = git_output("rev-parse", f"{EXPECTED_COMMIT}:{relative}")
            current_blob = git_output("hash-object", relative)
            matches = path.is_file() and frozen_blob == current_blob
        except (subprocess.CalledProcessError, FileNotFoundError):
            matches = False
        blob_matches[relative] = matches
        if not matches:
            errors.append(f"frozen artifact differs from bound commit: {relative}")

    raw_path = ROOT / "07_results/raw/e0_h039_results.json"
    table_path = ROOT / "07_results/tables/e0_h039_summary.csv"
    if raw_path.exists() or table_path.exists():
        errors.append("formal H-039 results existed at binding time")

    state = yaml.safe_load((ROOT / "research_state.yaml").read_text(encoding="utf-8"))
    if state.get("latest_decision", {}).get("decision_id") != "D-0024":
        errors.append("research_state does not record D-0024")
    if state.get("budget", {}).get("used_units") != 70:
        errors.append("binding changed the research budget")
    if set(state.get("branches", {}).get("active", [])) != {"H-001", "H-005", "H-014", "H-039"}:
        errors.append("binding changed the active portfolio")

    decision_log = (ROOT / "09_decisions/decision_log.md").read_text(encoding="utf-8")
    if "D-0024" not in decision_log or "BIND_H039_E0_CODE_SNAPSHOT" not in decision_log:
        errors.append("H-039 binding decision is missing")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    if "v0.12.2" not in readme or EXPECTED_COMMIT not in readme:
        errors.append("README lacks the H-039 binding summary")

    output = {
        "experiment_id": prereg.get("experiment_id"),
        "bound_commit": prereg.get("code_commit"),
        "frozen_blob_matches": blob_matches,
        "formal_results_absent": not raw_path.exists() and not table_path.exists(),
        "errors": errors,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
