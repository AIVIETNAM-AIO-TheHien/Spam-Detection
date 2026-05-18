# Kế hoạch Dự án Spam Detection V2

---

## 1. Phân định Vai trò và Trách nhiệm (Cập nhật)

| Vai trò | Trách nhiệm chính | Sản phẩm bàn giao |
| :--- | :--- | :--- |
| **AI Data** | Thu thập, làm giàu (Augmentation) và quy chuẩn hóa tiền xử lý. | Script preprocess_fixed.py, Custom Dataset (v2), Báo cáo giải trình Dữ liệu. |
| **AI Model** | Phân tích lỗi (Error Analysis) của Baseline Kaggle, chuyển đổi module và huấn luyện các mô hình bậc cao. | File model (.pkl), Báo cáo so sánh (Baseline Kaggle vs Custom Data & Models). |
| **AI Pipeline** | Đưa Baseline có sẵn lên luồng thử nghiệm, xây dựng giao diện Streamlit và quản lý tích hợp. | Source code Streamlit, Môi trường triển khai chuẩn (GitHub/Docker). |
| **QA/QC** | Tấn công Baseline hiện tại để tìm lỗ hổng, xây dựng bộ test độc lập. | Bộ Test-cases đặc thù, Báo cáo kiểm định chất lượng (QA Report). |

---

## 2. Kế hoạch Tuần 1: Đánh giá Baseline & Quy chuẩn hóa (11/05 - 17/05)

| Thành viên | Mục tiêu (Key Results) | Yêu cầu Kỹ thuật / Giải trình |
| :--- | :--- | :--- |
| **AI Data** | Quy chuẩn hóa Dataset hiện tại & Lên kịch bản làm giàu dữ liệu. | Thu thập data (bất động sản, lừa đảo trúng thưởng, khóa học, đa cấp...) và chỉ ra "khó khăn/điểm yếu". Xây dựng file preprocess_fixed.py chuẩn. |
| **AI Model** | Tái cấu trúc code Baseline & Phân tích lỗi (Error Analysis). | Refactor code Naive Bayes dùng thư viện preprocess_fixed.py. Trích xuất các mẫu bị đoán sai từ tập Kaggle để phân tích. |
| **AI Pipeline** | Đóng gói Baseline lên Streamlit. | Viết API/Streamlit kết nối trực tiếp với model Naive Bayes để nhóm có công cụ test ngay lập tức. |
| **QA/QC** | Kiểm thử hộp đen (Black-box) Baseline. | Nhập các câu spam "đời thực" vào Streamlit để chứng minh model Kaggle đang bị thiếu sót ở đâu. |

---

## 3. Kế hoạch Tuần 2: Nâng cấp Custom Dataset & Đào sâu Mô hình (18/05 - 24/05)

| Thành viên | Mục tiêu (Key Results) | Yêu cầu Kỹ thuật / Giải trình |
| :--- | :--- | :--- |
| **AI Data** | Phát hành Custom Dataset (v2) & Áp dụng DVC. | Sử dụng LLM sinh thêm data đắp vào chỗ hụt. Quản lý version qua Data Version Control (DVC). |
| **AI Model** | Huấn luyện & So sánh trên Data mới. | Huấn luyện lại Naive Bayes, bổ sung SVM/Logistic Regression trên Custom Dataset v2. Giải thích vì sao kết quả thay đổi. |
| **AI Pipeline** | Cập nhật luồng dữ liệu mới. | Đảm bảo Streamlit load model mới nhất và có thể tự động hóa các bước. |
| **QA/QC** | Nghiệm thu toàn hệ thống. | Stress-test. Đối chiếu kết quả dự đoán giữa tuần 1 và tuần 2 để làm báo cáo cải thiện. |

---

## 4. Tiêu chuẩn Kỹ thuật Bắt buộc

* **Tính mô-đun hóa**: File `preprocess_fixed.py` phải được import xuyên suốt, không được code cứng (hard-code) các bước làm sạch.
* **Báo cáo "Đào sâu" (Justification)**: Cần làm rõ điểm yếu của dữ liệu ban đầu và cách Custom Data khắc phục.
* **Quản lý Version**: Kiểm soát chặt sự đồng bộ giữa phiên bản Data và phiên bản Model.
