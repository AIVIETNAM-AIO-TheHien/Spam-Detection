# QA/QC — Tiêu chí Đánh giá & Checklist Nghiệm thu

> **Phiên bản:** 2.0 · **Cập nhật:** 11/05/2026  
> **Tham chiếu:** [Plan.md](Plan.md) — Kế hoạch Dự án Spam Detection V2

Tài liệu này định nghĩa **tiêu chí nghiệm thu (Acceptance Criteria)** cho từng vai trò trong dự án, chia theo tuần. Mỗi tiêu chí là một checkbox — đánh dấu `[x]` khi đạt, `[ ]` khi chưa đạt.

---

## Quy ước Mức độ Ưu tiên

| Ký hiệu | Ý nghĩa | Hậu quả nếu thiếu |
| :---: | :--- | :--- |
| 🔴 | **Bắt buộc (P0)** | Block cả nhóm, không thể nghiệm thu tuần. |
| 🟡 | **Quan trọng (P1)** | Ảnh hưởng chất lượng, cần hoàn thành trước khi sang tuần tiếp. |
| 🟢 | **Nên có (P2)** | Nâng cao chất lượng, có thể bổ sung sau. |

---

## TUẦN 1: Đánh giá Baseline & Quy chuẩn hóa (11/05 – 17/05)

### 1.1. AI Data

**Mục tiêu:** Quy chuẩn hóa Dataset hiện tại & Lên kịch bản làm giàu dữ liệu.

#### a) Thu thập & Quy chuẩn hóa Dữ liệu

- [x] 🔴 Dataset Kaggle gốc đã được lưu vào thư mục `data/` đúng định dạng (CSV/JSONL).
- [x] 🔴 Schema dữ liệu nhất quán: các cột được đặt tên rõ nghĩa, không có cột thừa/trống vô nghĩa.
- [ ] 🟡 Thu thập thêm mẫu spam "đời thực" tiếng Việt (bất động sản, lừa đảo trúng thưởng, khóa học, đa cấp...) và ghi chú nguồn gốc.
- [ ] 🟡 Tài liệu mô tả rõ **khó khăn/điểm yếu** của dataset hiện tại (thiếu domain nào, class nào ít mẫu...).

#### b) File `preprocess_fixed.py`

- [x] 🔴 File `preprocess_fixed.py` tồn tại, chạy được, có docstring giải thích từng hàm.
- [x] 🔴 Bao gồm tối thiểu các bước: lowercase, loại bỏ ký tự đặc biệt, tokenize.
- [x] 🟡 Có thể import và tái sử dụng từ bất kỳ script/notebook nào khác (`from src.preprocess import ...`).
- [x] 🟢 Có unit test cơ bản cho `preprocess_fixed.py` và tất cả test PASS.

#### c) Kịch bản Làm giàu Dữ liệu (Augmentation Plan)

- [x] 🟡 Tài liệu (hoặc section trong notebook) mô tả chiến lược augmentation: dùng LLM sinh data, back-translation, paraphrase, v.v.
- [x] 🟢 Ước tính số lượng mẫu cần bổ sung cho từng class/domain.

---

### 1.2. AI Model

**Mục tiêu:** Tái cấu trúc code Baseline & Phân tích lỗi (Error Analysis).

#### a) Refactor Baseline (Naive Bayes)

- [x] 🔴 Code Naive Bayes + TF-IDF chạy thành công trên dataset Kaggle.
- [x] 🔴 Code sử dụng `preprocess_fixed.py` của team Data (import trực tiếp, **không hard-code** bước tiền xử lý).
- [x] 🔴 Script đánh giá in ra đầy đủ các chỉ số: **Accuracy, F1-score, Precision, Recall**.
- [ ] 🟡 Code được tổ chức thành hàm/class rõ ràng, có docstring, tuân thủ PEP8.

#### b) Phân tích Lỗi (Error Analysis)

- [x] 🔴 Trích xuất danh sách các mẫu bị dự đoán sai (False Positive & False Negative).
- [x] 🟡 Phân loại/gom nhóm các mẫu sai theo đặc điểm (VD: quá ngắn, chứa URL, ngôn ngữ lẫn...).
- [x] 🟡 Tài liệu hoặc notebook tóm tắt pattern lỗi chính, đề xuất hướng cải thiện.
- [x] 🟢 Confusion matrix được trình bày dưới dạng bảng trong error analysis

#### c) Chiến lược Chia Dữ liệu

- [x] 🟡 Phân tách Train/Dev/Test rõ ràng, ghi lại seed hoặc lưu indices cố định.
- [x] 🟡 Tỷ lệ phân phối class trong mỗi tập gần với tập gốc (stratified split).

---

### 1.3. AI Pipeline

**Mục tiêu:** Đóng gói Baseline lên Streamlit.

#### a) Ứng dụng Streamlit

- [x] 🔴 App Streamlit chạy được (`streamlit run scripts/streamlit_app.py` không lỗi).
- [x] 🔴 Giao diện cho phép nhập văn bản → nhấn nút → hiển thị kết quả dự đoán (Spam/Ham).
- [ ] 🔴 Kết nối trực tiếp với model Naive Bayes đã train (load file `.pkl` hoặc gọi hàm predict).
- [x] 🟡 Giao diện hiển thị thêm xác suất dự đoán (probability/confidence score).
- [x] 🟢 Có hướng dẫn chạy app trong `README.md` hoặc `SETUP.md`.

#### b) Tích hợp Pipeline

- [x] 🟡 Luồng xử lý: Input text → `preprocess_fixed.py` → TF-IDF transform → Model predict → Output.
- [x] 🟡 Sử dụng đúng `preprocess_fixed.py` của team Data (không duplicate logic tiền xử lý).
- [x] 🟢 Có xử lý edge case: input rỗng, input quá dài, ký tự đặc biệt.

---

### 1.4. QA/QC

**Mục tiêu:** Kiểm thử hộp đen (Black-box) Baseline.

#### a) Bộ Test-cases Đặc thù

- [ ] 🔴 Chuẩn bị ≥ 20 câu spam "đời thực" (tiếng Việt/Anh) bao gồm các dạng: quảng cáo, lừa đảo, đa cấp, phishing link.
- [ ] 🔴 Chuẩn bị ≥ 10 câu ham (tin nhắn bình thường) để kiểm tra false positive.
- [ ] 🔴 Nhập từng câu vào Streamlit và ghi lại kết quả dự đoán (đúng/sai).

#### b) Báo cáo Kiểm thử Tuần 1

- [ ] 🔴 Tài liệu (hoặc bảng) thống kê: tổng số test, số đúng, số sai, tỷ lệ chính xác.
- [x] 🟡 Chỉ ra rõ **model đang thiếu sót ở đâu** (domain nào, pattern nào model hay sai).
- [x] 🟡 Đề xuất hướng khắc phục cụ thể cho Tuần 2 (VD: cần thêm data domain X, cần feature Y).
- [ ] 🟢 Lưu bộ test-cases thành file có cấu trúc (CSV/JSON) để tái sử dụng ở Tuần 2.

#### c) Review Chéo & Issue Log

- [x] 🟡 Review Problem Statement: bài toán đã được giới hạn phạm vi rõ ràng chưa?
- [x] 🟡 Review cấu trúc repo: thư mục đã tổ chức hợp lý chưa (`src/`, `data/`, `docs/`, `tests/`...)?
- [x] 🟡 Ghi nhận tất cả issue vào Issue Log (Google Sheet / GitHub Issues) kèm mức priority.

---

### Checklist Tổng hợp Tuần 1

| # | Hạng mục | Trạng thái |
| :---: | :--- | :---: |
| 1 | Dataset gốc đã chuẩn hóa và lưu đúng vị trí | ☑ |
| 2 | `preprocess_fixed.py` hoạt động, có thể import xuyên suốt | ☑ |
| 3 | Baseline Naive Bayes chạy thành công, có metrics rõ ràng | ☑ |
| 4 | Error Analysis: danh sách mẫu sai đã được trích xuất | ☑ |
| 5 | Streamlit app chạy được, nhập text → trả kết quả | ☑ |
| 6 | Bộ test black-box ≥ 30 câu đã thực hiện xong | ☐ |
| 7 | Báo cáo QA Tuần 1 đã hoàn thành | ☐ |
| 8 | Issue Log đã cập nhật | ☐ |

---

## TUẦN 2: Nâng cấp Custom Dataset & Đào sâu Mô hình (18/05 – 24/05)

### 2.1. AI Data

**Mục tiêu:** Phát hành Custom Dataset (v2) & Áp dụng DVC.

#### a) Custom Dataset v2

- [x] 🔴 Dataset mới đã được tạo, bao gồm dữ liệu gốc + dữ liệu sinh bởi LLM.
- [ ] 🔴 Dữ liệu LLM-generated được đánh dấu rõ ràng (cột `source` hoặc metadata).
- [x] 🔴 Dataset v2 đã bù đắp các lỗ hổng được QA/QC chỉ ra ở Tuần 1 (thêm domain thiếu, cân bằng class).
- [ ] 🟡 Chất lượng data sinh: QA xác nhận ≥ 90% mẫu sinh có ngữ nghĩa tự nhiên, không lặp lại máy móc.
- [ ] 🟡 Tổng số mẫu dataset v2 có sự cải thiện rõ rệt so với v1 (ghi rõ số liệu).

#### b) Quản lý Version với DVC

- [ ] 🟡 DVC đã được cài đặt và cấu hình trong repo (`dvc init`, `.dvc` files tồn tại).
- [ ] 🟡 Dataset v1 và v2 được track riêng biệt, có thể chuyển đổi qua lại (`dvc checkout`).
- [ ] 🟢 Tài liệu hướng dẫn sử dụng DVC cho thành viên mới.

#### c) Báo cáo Giải trình Dữ liệu

- [ ] 🔴 Tài liệu so sánh rõ ràng giữa Dataset v1 vs v2: số lượng mẫu, phân phối class, domain coverage.
- [ ] 🟡 Giải trình cụ thể: điểm yếu nào của v1 đã được khắc phục, bằng cách nào.

---

### 2.2. AI Model

**Mục tiêu:** Huấn luyện & So sánh trên Data mới.

#### a) Huấn luyện Mô hình Nâng cao

- [ ] 🔴 Huấn luyện lại **Naive Bayes** trên Custom Dataset v2 — ghi nhận metrics.
- [x] 🔴 Huấn luyện thêm ≥ 1 mô hình bổ sung (**SVM** hoặc **Logistic Regression**).
- [x] 🔴 Tất cả model sử dụng `preprocess_fixed.py` chung (không hard-code riêng).

#### b) Báo cáo So sánh

- [x] 🔴 Bảng so sánh metrics (Accuracy, F1, Precision, Recall) giữa:
  - Naive Bayes trên Kaggle (Tuần 1) vs Naive Bayes trên Custom v2.
  - Naive Bayes vs SVM vs Logistic Regression trên cùng Custom v2.
- [ ] 🔴 **Giải thích** vì sao kết quả thay đổi (data mới tốt hơn/xấu hơn ở điểm nào).
- [x] 🟡 Trực quan hóa so sánh bằng biểu đồ (bar chart metrics, confusion matrix so sánh).
- [x] 🟢 Phân tích lỗi (Error Analysis) trên model tốt nhất của Tuần 2.

#### c) Hyperparameter & Cross-validation

- [x] 🟡 Thử nghiệm tối ưu siêu tham số (GridSearch/RandomSearch) cho ≥ 1 model.
- [x] 🟡 Áp dụng Cross-validation (≥ 5-fold) trên tập Train để đánh giá ổn định.
- [x] 🟢 Ghi lại bảng kết quả hyperparameter search.

---

### 2.3. AI Pipeline

**Mục tiêu:** Cập nhật luồng dữ liệu mới.

#### a) Tích hợp Model Mới

- [x] 🔴 Streamlit app load được model mới nhất (Tuần 2) thay vì model Baseline Tuần 1.
- [x] 🔴 Có cơ chế chọn model để so sánh (VD: dropdown chọn giữa Naive Bayes / SVM / LR).
- [x] 🟡 Hiển thị tên model + version đang dùng trên giao diện.

#### b) Tự động hóa Pipeline

- [ ] 🟡 Script hoặc Makefile cho phép chạy toàn bộ pipeline E2E: `data → preprocess → train → evaluate`.
- [ ] 🟡 Pipeline tự động ghi log kết quả (metrics, timestamp) vào file.
- [ ] 🟢 Hỗ trợ truyền tham số dòng lệnh (chọn model, chọn dataset version).

#### c) Chất lượng Code

- [x] 🟡 Code mới tuân thủ PEP8, có docstring.
- [x] 🟡 Có unit test cho các module mới và tất cả test PASS.
- [x] 🟢 Không có import thừa, không có code chết (dead code).

---

### 2.4. QA/QC

**Mục tiêu:** Nghiệm thu toàn hệ thống.

#### a) Stress-test Hệ thống

- [ ] 🔴 Chạy lại toàn bộ bộ test-cases Tuần 1 trên model Tuần 2 — ghi nhận kết quả.
- [ ] 🔴 Bổ sung thêm ≥ 10 test-cases mới dựa trên các domain đã bổ sung ở v2.
- [ ] 🟡 Test edge cases nâng cao: tin nhắn rất ngắn (< 5 từ), rất dài, mix ngôn ngữ, chứa emoji/URL.
- [ ] 🟡 Đo thời gian phản hồi (response time) của Streamlit app.

#### b) Báo cáo Đối chiếu Tuần 1 vs Tuần 2

- [x] 🔴 Bảng so sánh kết quả: cùng bộ test → kết quả Tuần 1 vs Tuần 2 (có cải thiện không?).
- [x] 🔴 Kết luận rõ ràng: model/data Tuần 2 **có tốt hơn** Tuần 1 không, ở điểm nào.
- [x] 🟡 Phân tích các trường hợp Tuần 1 sai nhưng Tuần 2 vẫn sai — đề xuất hướng tiếp theo.

#### c) Nghiệm thu Kỹ thuật

- [x] 🟡 Kiểm tra `preprocess_fixed.py` được import xuyên suốt (không có file nào hard-code riêng).
- [x] 🟡 Kiểm tra đồng bộ version: Data version ↔ Model version ↔ Pipeline đang load đúng.
- [x] 🟡 Kiểm tra tính hoàn thiện tài liệu: README, SETUP, báo cáo các role đã nộp đủ.
- [x] 🟢 Code review: ≥ 1 pull request từ mỗi role đã được review và approve.

---

### Checklist Tổng hợp Tuần 2

| # | Hạng mục | Trạng thái |
| :---: | :--- | :---: |
| 1 | Custom Dataset v2 đã phát hành, có giải trình so sánh v1 vs v2 | ☐ |
| 2 | DVC đã cấu hình, track được v1 và v2 | ☐ |
| 3 | ≥ 2 model đã huấn luyện, có bảng so sánh metrics | ☑ |
| 4 | Báo cáo giải thích sự thay đổi kết quả | ☑ |
| 5 | Streamlit cập nhật model mới, có dropdown chọn model | ☑ |
| 6 | Pipeline E2E có thể chạy tự động | ☐ |
| 7 | Stress-test ≥ 40 test-cases đã hoàn thành | ☐ |
| 8 | Báo cáo đối chiếu Tuần 1 vs Tuần 2 đã nộp | ☐ |
| 9 | Version đồng bộ: Data ↔ Model ↔ Pipeline | ☑ |

---
-
