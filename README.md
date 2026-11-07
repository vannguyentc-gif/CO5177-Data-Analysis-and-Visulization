# CO5177 – Data Analysis and Visualization

Bài tập lớn cá nhân của **Nguyễn Thị Cẩm Vân – MSSV 2570803**, nhóm **BugIsFeature**.

## Nội dung

- **Tabular:** dự đoán xe "lemon" trong đấu giá xe cũ bằng Logistic Regression, Random Forest và HistGradientBoosting.
- **Time series:** dự báo nồng độ CO₂ Mauna Loa trước 12 tuần bằng baseline, Ridge, Random Forest và HistGradientBoosting.

## Xem bài làm

- Landing page: <https://vannguyentc-gif.github.io/CO5177-Data-Analysis-and-Visulization/>
- Báo cáo Tabular: `docs/tabular.html`
- Báo cáo Time series: `docs/timeseries.html`
- Notebook: `docs/notebooks/`
- PDF tổng hợp LMS: `BugIsFeature-report.pdf`

## Chạy lại

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe build_notebooks.py
```

Sau khi chạy hai notebook, tạo lại báo cáo và PDF:

```powershell
.\.venv\Scripts\python.exe build_reports.py
.\.venv\Scripts\python.exe export_pdfs.py
.\.venv\Scripts\python.exe validate_submission.py
```

GitHub Pages cần được cấu hình với nhánh `main`, thư mục `/docs`.