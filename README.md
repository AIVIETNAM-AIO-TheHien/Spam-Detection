# 🚀 Spam Detection Project - AIO Conquer 2026

Dự án phát triển hệ thống nhận diện thư và tin nhắn rác (Spam/Ham). Dự án này được thiết kế và quản lý theo tiêu chuẩn MLOps chuyên nghiệp để đảm bảo khả năng mở rộng và làm việc nhóm hiệu quả.

---

## 📁 Cấu trúc dự án (Project Structure)

Dự án được phân chia thư mục rõ ràng, tuyệt đối tuân thủ cấu trúc sau:

* `configs/`: Chứa các file cấu hình thông số (YAML/JSON). Mọi experiment phải được chạy dựa trên config từ đây, **không hardcode** thông số vào source code.
* `docs/`: Chứa tài liệu hướng dẫn, API docs, báo cáo dự án.
* `experiments/`: Nơi lưu lại logs, kết quả của các lần chạy thí nghiệm mô hình.
* `notebooks/`: Các file Jupyter Notebook dùng để explore data và phân tích thử nghiệm nhanh.
* `scripts/`: Chứa các file script chạy một lần (ví dụ: cào data, setup môi trường).
* `data/`: Chứa dataset gốc và dataset custom; không đặt file dữ liệu trong `src/`.
* `src/`: Thư mục chứa mã nguồn importable của dự án.
    * `data/`: Code tiền xử lý và validation dữ liệu.
    * `models/`: Helper/model classes trong tương lai, không chứa file `.joblib`.
    * `evaluation/`: Code chứa logic đánh giá mô hình.
    * `utils/`: Các hàm tiện ích dùng chung.
* `tests/`: Chứa các unit tests để kiểm tra code (bao gồm cả data tests).

---

## 👥 Quy trình làm việc của Team (Workflow)

Để đảm bảo source code luôn sạch và không bị lỗi vặt, toàn bộ team (5 thành viên) thống nhất tuân thủ quy trình làm việc trên GitHub như sau:

1.  **Bảo vệ nhánh chính (`main`)**: Nhánh `main` đã được khóa. **Nghiêm cấm và không thể** `git push` trực tiếp lên nhánh này.
2.  **Tạo nhánh mới (Branching)**: Mỗi khi làm một task mới (thêm model, sửa lỗi, xử lý data...), hãy tạo một branch mới từ `main`. 
    * *Ví dụ: `feature/lstm-model` hoặc `fix/data-bug`.*
3.  **Tạo Pull Request (PR)**: Sau khi code xong trên nhánh của mình, hãy đẩy lên GitHub và tạo một Pull Request. Hệ thống sẽ tự động hiện ra một bảng Checklist, yêu cầu điền đầy đủ thông tin.
4.  **Review và Merge**: PR cần ít nhất **1 lượt Approve** (từ Leader hoặc thành viên khác được chỉ định) mới có quyền gộp (Merge) vào nhánh `main`.
5.  **Báo cáo & Đề xuất**: Nếu gặp lỗi hoặc muốn đề xuất tính năng mới, hãy vào tab **Issues** trên GitHub và chọn đúng mẫu (`Bug report` hoặc `Feature request` trong folder .github ) để điền.

---

## ⚙️ Hướng dẫn cài đặt (Setup)

*(Phần này sẽ được cập nhật sau khi team chốt danh sách thư viện và dependencies)*
