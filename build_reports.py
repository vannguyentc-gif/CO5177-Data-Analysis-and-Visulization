"""Xuất notebook đã chạy thành báo cáo HTML (chỉ kết quả, ẩn code),
giao diện tương tự trang EDA mẫu của giảng viên.
Chạy: python build_reports.py
"""
import os, re
import nbformat
from nbconvert import HTMLExporter
from bs4 import BeautifulSoup

ROOT = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(ROOT, "docs", "notebooks")
OUT = os.path.join(ROOT, "docs")

REPORTS = [
    dict(nb="01_tabular_kick_lemon.ipynb", out="tabular.html",
         pdf="reports/tabular-report.pdf",
         icon="🚗", title="Kicked Car (Lemon) Dataset",
         subtitle="Báo cáo EDA & Machine Learning – Dữ liệu bảng (Tabular)",
         stats=[("72,983", "Samples"), ("33", "Features"), ("149,271", "Missing cells"),
                ("12.3%", "Lớp Lemon"), ("1:7.1", "Tỉ lệ mất cân bằng")]),
    dict(nb="02_timeseries_co2.ipynb", out="timeseries.html",
            pdf="reports/timeseries-report.pdf",
         icon="🌍", title="Mauna Loa CO₂ Dataset",
         subtitle="Báo cáo EDA & Dự báo – Dữ liệu chuỗi thời gian (Time series)",
         stats=[("2,284", "Điểm (tuần)"), ("1958–2001", "Khoảng thời gian"), ("59", "Giá trị thiếu"),
                ("12 tuần", "Horizon dự báo"), ("~1.3 ppm/năm", "Xu hướng")]),
]

CSS = """
<style>
:root{--brand:#1f3a5f;--brand2:#2e6da4;--bg:#eef2f7;--card:#fff;--line:#dde5ee}
body.report{margin:0;background:var(--bg);font-family:"Segoe UI",Roboto,Arial,sans-serif;color:#243040}
.hero{background:linear-gradient(135deg,#1f3a5f 0%,#2e6da4 100%);color:#fff;padding:46px 6% 34px;text-align:center}
.hero h1{margin:0;font-size:34px;font-weight:700}
.hero p.sub{margin:10px 0 4px;font-size:16px;opacity:.92}
.hero p.meta{margin:2px 0;font-size:13px;opacity:.78}
.stats{display:flex;flex-wrap:wrap;justify-content:center;gap:14px;margin-top:22px}
.stat{background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.25);border-radius:10px;padding:10px 18px;min-width:110px}
.stat b{display:block;font-size:20px} .stat span{font-size:12px;opacity:.85}
.toolbar{position:sticky;top:0;z-index:50;background:#fff;border-bottom:1px solid var(--line);padding:10px 6%;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.toolbar a,.toolbar button{font:inherit;font-size:13px;border:1px solid var(--brand);background:#fff;color:var(--brand);border-radius:16px;padding:5px 14px;cursor:pointer;text-decoration:none}
.toolbar a:hover,.toolbar button:hover{background:var(--brand);color:#fff}
.wrap{max-width:1180px;margin:26px auto 60px;padding:0 18px}
details.sec{background:var(--card);border:1px solid var(--line);border-radius:12px;margin-bottom:18px;box-shadow:0 1px 3px rgba(20,40,70,.07);overflow:hidden}
details.sec>summary{cursor:pointer;list-style:none;padding:16px 22px;font-size:19px;font-weight:600;color:var(--brand);background:#f7fAff;border-bottom:1px solid transparent}
details.sec[open]>summary{border-bottom:1px solid var(--line)}
details.sec>summary::-webkit-details-marker{display:none}
details.sec>summary::after{content:"▼";float:right;font-size:12px;opacity:.6;transition:.2s}
details.sec:not([open])>summary::after{transform:rotate(-90deg)}
.secbody{padding:8px 22px 22px}
.secbody h3{color:var(--brand2);border-left:4px solid var(--brand2);padding-left:10px;margin:22px 0 10px}
.secbody table{border-collapse:collapse;font-size:13px;margin:10px 0;max-width:100%;display:block;overflow-x:auto}
.secbody table th{background:var(--brand);color:#fff;font-weight:600;padding:7px 10px;text-align:left;position:sticky;top:0}
.secbody table td{padding:6px 10px;border-bottom:1px solid #eef1f5}
.secbody tbody tr:nth-child(even){background:#fafcff}
.secbody img{max-width:100%;height:auto;display:block;margin:12px auto;border-radius:6px}
.jp-OutputArea-output{background:transparent!important}
.jp-RenderedHTMLCommon p strong:first-child{color:#b3541e}
blockquote,.note{border-left:4px solid #f0a500;background:#fffaf0;padding:10px 14px;border-radius:0 8px 8px 0;margin:12px 0}
footer.rep{text-align:center;color:#5a6b80;font-size:13px;padding:26px 6% 40px;border-top:1px solid var(--line);background:#fff}
/* ẩn phần code (nếu còn), giữ nguyên markdown đã render */
.jp-CodeCell .jp-InputArea,.jp-CodeCell .jp-Cell-inputWrapper,
.jp-InputPrompt,.jp-OutputPrompt,.jp-Collapser{display:none!important}
@media print{.toolbar{display:none}details.sec{break-inside:avoid}}
</style>
"""

JS = """
<script>
document.addEventListener('DOMContentLoaded',function(){
  document.getElementById('btn-expand').onclick=()=>document.querySelectorAll('details.sec').forEach(d=>d.open=true);
  document.getElementById('btn-collapse').onclick=()=>document.querySelectorAll('details.sec').forEach(d=>d.open=false);
  document.querySelectorAll('.toolbar a[href^="#"]').forEach(a=>a.addEventListener('click',e=>{
    const t=document.querySelector(a.getAttribute('href')); if(t){t.open=true;}
  }));
});
</script>
"""


def slug(s, i):
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return f"sec-{i}-{s[:40]}" if s else f"sec-{i}"


def build(cfg):
    nb = nbformat.read(os.path.join(NB, cfg["nb"]), as_version=4)
    exporter = HTMLExporter(template_name="lab", exclude_input=True,
                            exclude_input_prompt=True, exclude_output_prompt=True)
    body, _ = exporter.from_notebook_node(nb)
    soup = BeautifulSoup(body, "html.parser")

    container = soup.select_one("#notebook-container") or soup.find("body")
    cells = container.find_all("div", class_="jp-Cell", recursive=False) or \
        [c for c in container.find_all("div", class_="jp-Cell")]

    # Gom các cell thành section theo tiêu đề h2
    sections, cur = [], None
    for c in cells:
        for h1 in c.find_all("h1"):      # tiêu đề chính đã nằm ở hero
            h1.decompose()
        for a in c.select("a.anchor-link"):
            a.decompose()
        h2 = c.find("h2")
        if h2:
            cur = {"title": h2.get_text(" ", strip=True), "nodes": []}
            h2.decompose()
            sections.append(cur)
        if cur is None:
            cur = {"title": "Giới thiệu", "nodes": []}
            sections.append(cur)
        # bỏ qua cell code không có output (vì code đã bị ẩn)
        if "jp-CodeCell" in (c.get("class") or []) and not c.select_one(".jp-OutputArea-output"):
            continue
        cur["nodes"].append(c)

    out = BeautifulSoup("<!DOCTYPE html><html lang='vi'><head></head><body class='report'></body></html>",
                        "html.parser")
    head, bd = out.head, out.body
    head.append(BeautifulSoup('<meta charset="utf-8">'
                              '<meta name="viewport" content="width=device-width, initial-scale=1">'
                              f'<title>{cfg["title"]} – Báo cáo</title>', "html.parser"))
    # giữ style + script của nbconvert (plotly, MathJax...)
    for tag in soup.find_all(["style", "script", "link"]):
        head.append(tag.extract())
    head.append(BeautifulSoup(CSS, "html.parser"))

    stats = "".join(f'<div class="stat"><b>{v}</b><span>{k}</span></div>' for v, k in cfg["stats"])
    bd.append(BeautifulSoup(f"""
<div class="hero">
  <h1>{cfg['icon']} {cfg['title']}</h1>
  <p class="sub">{cfg['subtitle']}</p>
  <p class="meta">Nền tảng lập trình cho phân tích và trực quan dữ liệu · HK 261, 2026–2027 · GV: Lê Thành Sách</p>
  <p class="meta">ĐH Bách Khoa – ĐHQG-HCM · Khoa Khoa học và Kỹ thuật Máy tính</p>
    <p class="meta"><strong>Nhóm BugIsFeature · Nguyễn Thị Cẩm Vân · MSSV 2570803</strong></p>
  <div class="stats">{stats}</div>
</div>""", "html.parser"))

    ids = [slug(s["title"], i) for i, s in enumerate(sections)]
    nav = "".join(f'<a href="#{i}">{s["title"]}</a>' for i, s in zip(ids, sections))
    bd.append(BeautifulSoup(
        f'<div class="toolbar"><a href="index.html">🏠 Trang chính</a>'
        f'<a href="notebooks/{cfg["nb"]}">📓 Notebook</a>'
        f'<a href="{cfg["pdf"]}">📄 PDF report</a>'
        f'<button id="btn-expand">📂 Mở tất cả</button><button id="btn-collapse">📦 Thu gọn</button>{nav}</div>',
        "html.parser"))

    wrap = out.new_tag("div", attrs={"class": "wrap"})
    bd.append(wrap)
    for sid, s in zip(ids, sections):
        d = out.new_tag("details", attrs={"class": "sec", "id": sid})
        d["open"] = ""
        sm = out.new_tag("summary"); sm.string = s["title"]; d.append(sm)
        inner = out.new_tag("div", attrs={"class": "secbody"})
        for n in s["nodes"]:
            inner.append(n)
        d.append(inner); wrap.append(d)

    bd.append(BeautifulSoup(
        f'<footer class="rep">Nhóm BugIsFeature · Nguyễn Thị Cẩm Vân · MSSV 2570803<br>'
        f'{cfg["title"]} – báo cáo sinh tự động từ notebook đã thực thi.<br>'
        'Bài tập lớn môn Nền tảng lập trình cho phân tích và trực quan dữ liệu.</footer>' + JS,
        "html.parser"))

    path = os.path.join(OUT, cfg["out"])
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(out))
    print("wrote", path, f"({len(sections)} sections)")


if __name__ == "__main__":
    for c in REPORTS:
        build(c)
