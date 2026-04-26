# 🛡️ Spam Detection Project - AIO Conquer 2026

Dự án nhận diện thư/tin nhắn rác (Spam/Ham) trong thời gian 4 tuần.

## 📁 Cấu trúc thư mục (Project Structure)

Dự án được tổ chức theo chuẩn MLOps chuyên nghiệp:

* [cite_start]**`configs/`**: Chứa các file cấu hình thông số (YAML/JSON)[cite: 383]. [cite_start]Mọi experiment chạy từ đây, không hardcode[cite: 388].
* [cite_start]**`docs/`**: Chứa tài liệu hướng dẫn, API docs, báo cáo dự án[cite: 383].
* [cite_start]**`experiments/`**: Nơi lưu lại logs, kết quả của các lần chạy thí nghiệm mô hình[cite: 383].
* [cite_start]**`notebooks/`**: Các file Jupyter Notebook dùng để explore data, phân tích thử nghiệm nhanh[cite: 383].
* [cite_start]**`scripts/`**: Chứa các file script chạy một lần (ví dụ: cào data, setup môi trường)[cite: 383].
* [cite_start]**`src/`**: Thư mục chứa mã nguồn chính của dự án[cite: 383].
    * [cite_start]`data/`: Code tiền xử lý và nạp dữ liệu[cite: 383].
    * [cite_start]`models/`: Định nghĩa các kiến trúc mô hình (Baseline, LSTM, Transformer...)[cite: 383].
    * [cite_start]`training/`: Luồng huấn luyện mô hình (Training loops)[cite: 383].
    * [cite_start]`evaluation/`: Code chứa logic đánh giá mô hình[cite: 383].
    * [cite_start]`utils/`: Các hàm tiện ích dùng chung[cite: 383].
* [cite_start]**`tests/`**: Chứa các unit tests để kiểm tra code (bao gồm cả data tests)[cite: 383].

## 🚀 Hướng dẫn cài đặt (Setup)
*(Sẽ cập nhật sau khi team chốt danh sách thư viện)*
