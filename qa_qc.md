# QA/QC Week 01 - Tiêu chí Đánh giá & Checklist

Tài liệu này định nghĩa các nhiệm vụ của role QA/QC trong Tuần 1 và các tiêu chí cụ thể để đánh giá thành quả công việc (deliverables) của các role khác trong dự án Spam Detection.

## 1. Nhiệm vụ QA/QC Tuần 1
- **Làm rõ tiêu chí chất lượng:** Định nghĩa F1-score mục tiêu, cách đánh giá output của pipeline, và các tiêu chuẩn code review cơ bản.
- **Xây dựng checklist:** Lập checklist kiểm tra cho từng phần việc trong dự án (Problem statement, cấu trúc thư mục repo, thông tin EDA).
- **Review và theo dõi Issue:** Đánh giá lại Problem Statement và Sơ đồ pipeline, ghi nhận issue vào log và theo dõi mức độ ưu tiên.

## 2. Tiêu chí Đánh giá Thành quả của các Role (Week 01)

Dưới đây là các tiêu chí nghiệm thu (Acceptance Criteria) cụ thể dùng để đánh giá kết quả công việc của từng role dựa trên Roadmap của Tuần 1.

### 2.1. Đánh giá AIE Data
**Mục tiêu:** Thu thập, thực hiện EDA và chuẩn hóa dữ liệu.
- **Dataset thô:**
  - [ ] Số lượng sample thu thập được $\ge 5,000$.
  - [ ] Dữ liệu được lưu đúng định dạng JSONL hoặc CSV theo quy chuẩn.
  - [ ] Lược đồ (schema) dữ liệu nhất quán, các field được đặt tên hợp lý, rõ nghĩa.
- **Notebook EDA (`.ipynb`):**
  - [ ] Tính tái lập (Reproducibility): Chạy từ đầu đến cuối không báo lỗi.
  - [ ] Trực quan hóa đầy đủ: Có biểu đồ phân phối class (Spam/Ham), phân phối độ dài văn bản, WordCloud hoặc bar chart cho top từ phổ biến của cả Spam và Ham.
  - [ ] Khám phá dữ liệu: Đoạn code kiểm tra và báo cáo rõ về mức độ mất cân bằng (class imbalance), tỷ lệ dữ liệu trùng lặp (duplicate) và missing values.
- **Báo cáo Data Card:**
  - [ ] Tài liệu tóm tắt rõ nguồn gốc dữ liệu, phương pháp thu thập, các đặc trưng và cảnh báo về chất lượng dữ liệu (nếu có).

### 2.2. Đánh giá AIE Pipeline
**Mục tiêu:** Thiết kế pipeline và xây dựng các module cơ bản (Ingestion, Validation).
- **Sơ đồ Pipeline:**
  - [ ] Hình ảnh hoặc tài liệu diagram rõ ràng (qua draw.io, mermaid hoặc Markdown diagram).
  - [ ] Che phủ đầy đủ luồng End-to-End: Input → Ingestion → Validation → Preprocessing → Feature Engineering → Model → Evaluation → Output.
- **Tài liệu Spec Input/Output:**
  - [ ] Xác định rõ kiểu dữ liệu (data types), định dạng (format), và ràng buộc (null/not null, min/max length) tại mỗi bước chuyển tiếp.
- **Module Code (`ingestion.py`, `validation.py`):**
  - [ ] Tuân thủ chuẩn coding (VD: PEP8 đối với Python), có docstring giải thích tham số rõ ràng.
  - [ ] Xử lý được các lỗi cơ bản (Exception handling) khi đọc file hoặc validation thất bại.
  - [ ] Có Unit test cơ bản cho cả 2 module và các test này phải PASS.

### 2.3. Đánh giá AIE Model
**Mục tiêu:** Đánh giá các phương pháp tiếp cận, thiết lập baseline và chiến lược phân chia dữ liệu.
- **Model Selection Report:**
  - [ ] Tài liệu so sánh rõ ràng các khía cạnh ưu/nhược điểm, chi phí triển khai của: Rule-based, Classical ML (Naive Bayes, Logistic Regression, SVM) và Pre-trained LLM fine-tuning.
  - [ ] Đưa ra được kết luận và thứ tự ưu tiên thử nghiệm cho dự án.
- **Chiến lược Dev/Test Set:**
  - [ ] Chia tách tập dữ liệu đảm bảo nguyên tắc phân phối giống với thực tế chạy Production (cùng distribution).
  - [ ] Các chỉ mục (indices) của tập Dev/Test phải được lưu lại thành file cố định (ví dụ `test_indices.csv` hoặc seed cố định cứng) để đảm bảo không bị xáo trộn khi chạy lại nhiều lần.
- **Baseline Model:**
  - [ ] Chạy thành công mô hình cơ bản (Naive Bayes + TF-IDF) trên toàn bộ dữ liệu thô.
  - [ ] Script đánh giá in ra các chỉ số rõ ràng: Accuracy, F1-score (bắt buộc), Precision, Recall.

## 3. Checklist QA/QC Dự án (Week 1)
- [ ] **Problem Statement:** Đã được định nghĩa rõ ràng, giới hạn phạm vi bài toán chưa?
- [ ] **Repo Structure:** Cấu trúc thư mục mã nguồn đã đúng chuẩn chưa (ví dụ: chia src, data, notebooks, tests, Instructions...)?
- [ ] **Data Quality:** Thông tin EDA từ nhóm Data đã đủ sâu và sát với bài toán phân loại chưa?
- [ ] **Review Issue Log:** Các lỗi hoặc thiếu sót đã được QA/QC note lại, assign cho thành viên liên quan và đánh giá mức độ priority chưa?
