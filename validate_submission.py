"""Validate the assignment deliverables against assignment-individual.md."""

import json
import re
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"


def result(ok: bool, message: str) -> bool:
    print(f"[{'OK' if ok else 'MISSING'}] {message}")
    return ok


def notebook_ok(path: Path) -> bool:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    executed = sum(cell.get("execution_count") is not None for cell in code_cells)
    has_markdown = any(cell["cell_type"] == "markdown" for cell in notebook["cells"])
    return result(not errors and executed == len(code_cells) and has_markdown,
                  f"{path.name}: {executed}/{len(code_cells)} code cells executed, {len(errors)} errors")


def html_ok(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    body = text[text.find("<body"):]
    no_visible_code = "highlight hl-ipython3" not in body
    has_results = "<table" in body and ("data:image/png" in body or "plotly-graph-div" in body)
    has_identity = all(value in body for value in ("BugIsFeature", "Nguyễn Thị Cẩm Vân", "2570803"))
    return result(no_visible_code and has_results and has_identity,
                  f"{path.name}: result-only report, charts/tables and identity present")


def main() -> None:
    checks = []
    required = [
        ROOT / "data" / "kick.csv",
        ROOT / "data" / "co2_mauna_loa.csv",
        DOCS / "index.html",
        DOCS / "tabular.html",
        DOCS / "timeseries.html",
        DOCS / "notebooks" / "01_tabular_kick_lemon.ipynb",
        DOCS / "notebooks" / "02_timeseries_co2.ipynb",
        DOCS / "reports" / "tabular-report.pdf",
        DOCS / "reports" / "timeseries-report.pdf",
        DOCS / "reports" / "BugIsFeature-report.pdf",
        ROOT / "BugIsFeature-report.pdf",
    ]
    for path in required:
        checks.append(result(path.exists() and path.stat().st_size > 0, str(path.relative_to(ROOT))))

    checks.append(notebook_ok(DOCS / "notebooks" / "01_tabular_kick_lemon.ipynb"))
    checks.append(notebook_ok(DOCS / "notebooks" / "02_timeseries_co2.ipynb"))
    checks.append(html_ok(DOCS / "tabular.html"))
    checks.append(html_ok(DOCS / "timeseries.html"))

    for pdf in (DOCS / "reports" / "tabular-report.pdf",
                DOCS / "reports" / "timeseries-report.pdf",
                ROOT / "BugIsFeature-report.pdf"):
        pages = len(PdfReader(pdf).pages)
        checks.append(result(pages > 0, f"{pdf.name}: {pages} pages"))

    index = (DOCS / "index.html").read_text(encoding="utf-8")
    placeholders = re.findall(r"USERNAME|REPO|\[link\]|\[Họ tên\]|\[MSSV\]", index)
    checks.append(result(not placeholders, "Landing page has no technical placeholders"))
    checks.append(result("Video YouTube: chưa cập nhật" not in index,
                         "Two YouTube video links are published"))
    checks.append(result((ROOT / ".git").exists(), "Local Git repository is initialized"))

    print(f"\nResult: {sum(checks)}/{len(checks)} checks passed")
    raise SystemExit(0 if all(checks) else 1)


if __name__ == "__main__":
    main()