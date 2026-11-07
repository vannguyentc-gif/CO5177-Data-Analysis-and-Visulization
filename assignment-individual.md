# BÀI TẬP LỚN – Nền tảng lập trình cho phân tích và trực quan dữ liệu

- **Trường:** ĐH Bách Khoa – ĐHQG-HCM | **Khoa:** Khoa học và Kỹ thuật Máy tính
- **Học kỳ:** 261, năm học 2026–2027 | **Giảng viên:** Lê Thành Sách
- **Phiên bản tài liệu gốc:** v4.0 (14/09/2026)
- **Hình thức của bạn:** Sinh viên **cá nhân** → 2 bài: **Tabular (bắt buộc)** + **Time series (tự chọn)**

---

## 1. Mục tiêu
- Vận dụng Python, OOP, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn.
- Thực hiện quy trình: thu thập → tiền xử lý → EDA → trực quan hóa → huấn luyện mô hình.
- Làm quen nhiều loại dữ liệu (ở đây: Tabular, Time series).

## 2. Tài liệu tham khảo
- Mẫu EDA tabular (giao diện báo cáo cần làm tương tự): https://ltsach.github.io/AILearningHub/01_Data_Analysis/01_EDA/adult_income_tabular/eda_report_tabular.html
- GitHub của giảng viên: các ví dụ EDA, Machine Learning, Text/Image Classification (https://ltsach.github.io/AILearningHub/).
- Tệp đăng ký nhóm & landing page: lấy link trên LMS.

## 3. Khối lượng công việc (cá nhân)
| # | Loại dữ liệu | Mức độ |
|---|---|---|
| 1 | Tabular | Bắt buộc |
| 2 | Time series | Tự chọn |

- Khuyến khích dùng mô hình cơ bản của Scikit-learn. Nếu dùng Deep Learning: có thể dùng code mẫu, coi như hàm có sẵn.
- Tên nhóm (kể cả cá nhân) phải **không trùng** và điền vào sheet **GroupRegistration**; link landing page điền vào sheet **GroupLink**.

## 4. Ràng buộc dataset

### 4.1 Nguyên tắc chung
- Hạn chế chọn trùng dataset. Nếu trùng sẽ chấm song song và so sánh độ sáng tạo/khác biệt cách tiếp cận.

### 4.2 Tabular
- [ ] ≥ **2000 samples**, ≥ **10 cột**
- [ ] Có **missing values** (để thống kê và xử lý)
- [ ] Có cột **categorical** (encoding)
- [ ] Có cột **numerical** (scaling)
- [ ] Ưu tiên khó: imbalanced, outliers, …

### 4.3 Time series (tự chọn)
Tài liệu gốc không nêu ràng buộc riêng. Gợi ý nên đảm bảo:
- Có cột thời gian rõ ràng, đủ dài (vài trăm–nghìn điểm), có xu hướng/mùa vụ.
- **Chia train/val/test theo thời gian** (không shuffle, tránh data leakage).
- Có biểu đồ **dự báo so với thực tế**.

## 5. Cấu trúc nội dung mỗi notebook (`.ipynb`, chạy trên Google Colab)
Mỗi loại dữ liệu = 1 file notebook riêng.

1. **Mô tả tập dữ liệu**
   - Markdown mô tả tổng quan (nguồn, ý nghĩa cột, bài toán).
   - Trực quan bằng Plotly / Matplotlib / Seaborn.
2. **Chuẩn bị dữ liệu**
   - Tiền xử lý bằng NumPy/Pandas.
   - Xử lý missing values, outliers.
   - Encoding categorical; Scaling numerical.
   - Chia train / validation / test.
3. **Phân tích và mô hình hóa**
   - Huấn luyện bằng Scikit-learn.
   - So sánh kết quả giữa các mô hình/tham số.
4. **Tổng kết và mở rộng**
   - Đánh giá: accuracy, biểu đồ, confusion matrix, …
   - Nhận xét, hạn chế, đề xuất hướng mở rộng.

> Yêu cầu: **Run all** không lỗi trên Colab; có Markdown giải thích quy trình, biểu đồ, kết quả.

## 6. Quy cách nộp bài
Bài làm công bố trên **GitHub Pages**. Link landing page ghi vào sheet **GroupLink**. Notebook, video, file khác được link từ landing page.

### 6.1 Landing page
**Thông tin môn học**
- Trường: ĐH Bách Khoa – ĐHQG-HCM
- Khoa: Khoa học và Kỹ thuật Máy tính
- Học phần: Nền tảng lập trình cho phân tích và trực quan dữ liệu
- Học kỳ 261, năm học 2026–2027
- Giảng viên: Lê Thành Sách

**Thông tin nhóm**
- Tên nhóm (trùng sheet GroupRegistration)
- Thành viên: họ tên, MSSV, vai trò/đóng góp
- Link GitHub cá nhân (nếu có, không tạo link giả)
- Link repository

**Mục lục bài con**: link rõ ràng tới trang Tabular và Time series (mỗi bài một trang/mục riêng, không gộp một trang dài).

### 6.2 Trang mỗi bài con
- [ ] Tên bài và loại dữ liệu
- [ ] Tên nhóm và danh sách thành viên
- [ ] Mô tả bài toán và dataset
- [ ] Link notebook (GitHub và/hoặc Colab chạy được Run all)
- [ ] Link PDF report của bài đó
- [ ] Link video YouTube

### 6.3 PDF report và video
- Mỗi bài con: 1 PDF report (cách làm + kết quả) và 1 video (khuyến nghị 5–10 phút).
- Video **Public** hoặc **Unlisted**; tự kiểm tra link trước khi nộp.
- Tiêu đề gợi ý: `[Tên nhóm] – [Loại dữ liệu] – Nền tảng lập trình`.
- Cuối khóa nộp lên LMS PDF tổng hợp: `<groupname>-report.pdf` (ví dụ `DataViz-report.pdf`). Tên trùng sheet GroupRegistration. PDF LMS không thay landing page và video.

### 6.4 Notebook
- Mỗi loại dữ liệu một `.ipynb`, Run all không lỗi trên Colab, có Markdown giải thích.

### 6.5 Hạn nộp
- Landing page, trang bài con (kèm PDF, video) và PDF trên LMS: đúng hạn cuối khóa.
- Link landing page điền vào GroupLink trước hạn đăng ký do giảng viên thông báo.

## 7. Thang điểm
| Thành phần | Trọng số |
|---|---|
| Chất lượng và hoàn thiện notebook | 50% |
| Báo cáo, giải thích, trình bày Markdown | 20% |
| Video demo và khả năng trình bày | 15% |
| Sáng tạo, mở rộng, trình bày có kết luận | 10% |
| Hình thức nộp (GitHub Pages, landing page, PDF + video, PDF LMS) | 5% |
| **Tổng** | **100%** |

### 7.1 Cách lấy 10 điểm sáng tạo
Không thay thế phần bắt buộc; Run all là bắt buộc và không cộng vào cột này.
Được điểm khi hình/bảng **trả lời một câu hỏi**, caption **ghi kết luận**, có **số liệu đối chiếu**. Không cộng điểm cho việc đổi màu, word cloud, Plotly cho đẹp mà không nêu được điều gì.

| Mức | Điều kiện |
|---|---|
| 0 | Chỉ có biểu đồ bắt buộc (phân phối, missing, cân bằng lớp, confusion matrix), không thí nghiệm thêm |
| 4 | Mỗi loại dữ liệu có ≥ 1 hình đúng câu hỏi, caption nêu kết luận |
| 7 | Hình đó có số so sánh (trước/sau xử lý, mô hình A vs B, mẫu đúng vs sai) |
| 10 | Mỗi loại dữ liệu đạt mức 7 và kết luận khớp bảng metric trong report |

**Gợi ý cho 2 bài của bạn**
- **Tabular:** target vs 2–3 feature; phân phối trước/sau scale; baseline vs mô hình mở rộng (imbalance, outlier, feature importance).
- **Time series:** split theo thời gian; dự báo so với thực tế.

Điểm sáng tạo khác (tự thu thập dữ liệu, cùng dataset nhưng cách tiếp cận khác có số đối chiếu) vẫn được tính nếu chạy được và giải thích được. Đề xuất "có thể làm sau" mà không chạy thì **không** cộng điểm.

---

## 8. Checklist thực hiện

### Chung
- [ ] Đăng ký tên nhóm (GroupRegistration) và link landing page (GroupLink)
- [ ] Tạo repo + GitHub Pages, landing page đủ thông tin (mục 6.1)
- [ ] Giao diện báo cáo/EDA trực quan tương tự trang mẫu
- [ ] PDF tổng hợp `<groupname>-report.pdf` nộp LMS

### Bài 1 – Tabular
- [ ] Chọn dataset (≥2000 mẫu, ≥10 cột, missing, categorical, numerical, imbalanced/outlier)
- [ ] Mô tả + EDA (phân phối, missing, cân bằng lớp, tương quan)
- [ ] Xử lý missing/outlier, encoding, scaling, chia train/val/test
- [ ] Ít nhất 2–3 mô hình sklearn, so sánh metric, confusion matrix
- [ ] Thí nghiệm mở rộng (mức 7+): baseline vs xử lý imbalance/outlier, feature importance
- [ ] Kết luận, hạn chế, hướng mở rộng
- [ ] Notebook Run all trên Colab, PDF, video, trang bài con

### Bài 2 – Time series
- [ ] Chọn dataset có cột thời gian, đủ dài
- [ ] EDA: xu hướng, mùa vụ, missing, outlier
- [ ] Feature (lag, rolling, calendar), chia theo thời gian
- [ ] Baseline (naive) vs mô hình sklearn, metric MAE/RMSE/MAPE
- [ ] Biểu đồ dự báo vs thực tế có caption kết luận
- [ ] Kết luận, hạn chế, hướng mở rộng
- [ ] Notebook Run all trên Colab, PDF, video, trang bài con
