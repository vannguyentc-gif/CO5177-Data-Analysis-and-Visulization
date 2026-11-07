"""Export HTML reports to PDF and build the combined LMS submission."""

import base64
import shutil
import tempfile
import time
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from selenium import webdriver
from selenium.webdriver.chrome.options import Options


ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "docs" / "reports"
TARGETS = [
    (ROOT / "docs" / "index.html", REPORTS / "landing-page.pdf"),
    (ROOT / "docs" / "tabular.html", REPORTS / "tabular-report.pdf"),
    (ROOT / "docs" / "timeseries.html", REPORTS / "timeseries-report.pdf"),
]


def export_pdf(driver: webdriver.Chrome, html: Path, pdf: Path) -> None:
    driver.get(html.as_uri())
    driver.execute_script("document.querySelectorAll('details').forEach(d => d.open = true)")
    time.sleep(4)
    result = driver.execute_cdp_cmd(
        "Page.printToPDF",
        {
            "printBackground": True,
            "preferCSSPageSize": True,
            "paperWidth": 8.27,
            "paperHeight": 11.69,
            "marginTop": 0.35,
            "marginBottom": 0.35,
            "marginLeft": 0.35,
            "marginRight": 0.35,
        },
    )
    pdf.write_bytes(base64.b64decode(result["data"]))
    print(f"{pdf.name}: {len(PdfReader(pdf).pages)} pages, {pdf.stat().st_size:,} bytes")


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    profile = Path(tempfile.mkdtemp(prefix="assignment-chrome-"))
    options = Options()
    options.binary_location = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--allow-file-access-from-files")
    options.add_argument(f"--user-data-dir={profile}")

    try:
        driver = webdriver.Chrome(options=options)
        try:
            for html, pdf in TARGETS:
                export_pdf(driver, html, pdf)
        finally:
            driver.quit()
    finally:
        shutil.rmtree(profile, ignore_errors=True)

    writer = PdfWriter()
    for _, pdf in TARGETS:
        writer.append(str(pdf))
    combined = ROOT / "BugIsFeature-report.pdf"
    with combined.open("wb") as stream:
        writer.write(stream)
    shutil.copy2(combined, REPORTS / combined.name)
    print(f"{combined.name}: {len(PdfReader(combined).pages)} pages, {combined.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()