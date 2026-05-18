# Báo cáo Phản hồi: Quy chuẩn hóa cấu trúc đầu ra Preprocess Data

**Gửi:** Nhóm AI Data
**Dự án:** Spam Detection V2
**Mục tiêu:** Thống nhất định dạng đầu ra để trực tiếp đưa vào Model (Train/Test).

---

## 1. Yêu cầu về Định dạng Đầu ra (Format)
Để đảm bảo tính nhất quán với bộ dữ liệu Baseline (Kaggle SMS), yêu cầu mọi dữ liệu sau khi qua file `preprocess_fixed.py` (dù là email thô hay dữ liệu cào từ web) phải được trả về dưới dạng một dòng duy nhất theo cấu trúc:
`"nội dung văn bản đã làm sạch",label`

---

## 2. Các hành động yêu cầu trong script `preprocess_fixed.py`

Để đạt được "một dòng text duy nhất" mà không làm mất thông tin quan trọng của Email, AI Data cần thực hiện các thao tác sau:

### 2.1. Hợp nhất Subject và Body (Feature Fusion)
* **Vấn đề:** Email thô có Subject và Body tách biệt, trong khi Model Baseline chỉ nhận 1 trường text.
* **Giải pháp:** Thực hiện nối: `$Text_Duy_Nhat = Subject + " " + Body$`. 
* **Lưu ý:** Phải nối trước khi thực hiện các bước xóa stopword để đảm bảo tính liên kết ngôn ngữ.

### 2.2. Loại bỏ nhiễu hệ thống (Noise Stripping)
* **Vấn đề:** Các email thô (như tập Enron) chứa rất nhiều rác kỹ thuật (`_exmh_...`, mã PGP, header). Nếu để nguyên, dòng text duy nhất sẽ bị loãng bởi các ký tự vô nghĩa.
* **Hành động:** Sử dụng Regex để xóa triệt để:
    * Các ID phiên làm việc (ví dụ: `_exmh_1317289252p`).
    * Toàn bộ các khối chữ ký số (`begin pgp signature` ... `end pgp signature`).
    * Các từ khóa Header (`contenttype`, `charset`, `date`, `from`).
    * Toàn bộ chuỗi mã hóa Base64 / khối MIME attachment bị rò rỉ vào nội dung text (nhận dạng bằng chuỗi ký tự liên tục không có khoảng trắng dài hơn 50 ký tự).

### 2.3. Chuyển đổi thực thể thành Token (Entity Tokenization)
Để dòng text duy nhất mang tính tổng quát hóa cao, không bị phụ thuộc vào dữ liệu cũ (năm 2002), yêu cầu thay thế các giá trị cụ thể bằng nhãn đại diện:
* Số điện thoại -> `[PHONE]`
* Đường dẫn URL -> `[URL]`
* Địa chỉ Email -> `[EMAIL]`
* Số tiền/Tiền tệ -> `[MONEY]`

### 2.4. Xóa Boilerplate Footer Mailing List
* **Vấn đề:** Nhiều email trong dataset (Assassin/Enron) kết thúc bằng các đoạn footer lặp đi lặp lại có nội dung hoàn toàn không liên quan đến nội dung chính, ví dụ:
    * `irish linux users group for unsubscription information list maintainer`
    * `yahoo groups sponsor 4 dvds free sp join now to unsubscribe from this group send an email to...`
* **Vấn đề nếu bỏ qua:** Đây **không phải** header kỹ thuật nên regex trong mục 2.2 sẽ không bắt được. Các cụm từ này sẽ bị TF-IDF học nhầm thành feature phân biệt spam/ham.
* **Hành động:** Xây dựng một danh sách (blocklist) các pattern boilerplate phổ biến và dùng regex để xóa chúng trước khi xuất dòng text cuối cùng. Ưu tiên match theo pattern thay vì chuỗi cứng để bắt được các biến thể.

### 2.5. Kiểm soát Độ dài Văn bản (Text Length)
* **Vấn đề:** Sau khi hợp nhất Subject + Body (mục 2.1) và loại bỏ noise, một số email vẫn có thể dài hàng nghìn từ. Điều này không ảnh hưởng đến model TF-IDF + Naive Bayes hiện tại, nhưng sẽ gây vấn đề nếu về sau chuyển sang mô hình transformer (giới hạn 512 tokens).
* **Hành động đề xuất:** Sau khi preprocess, thêm bước thống kê và báo cáo phân bố độ dài văn bản (min/max/mean/p95) vào log.
* **Hành động đề xuất:** Truncate văn bản ở ngưỡng **1024 từ** (giữ phần đầu), đồng thời ghi chú rõ ngưỡng này trong config để dễ thay đổi sau.

---

## 3. Ví dụ minh họa quá trình biến đổi

### Ví dụ 1 — SMS spam (Kaggle baseline)
**Dữ liệu thô (Raw):**
> `Subject: Win $500! Body: Call 090xxx or visit http://spam.com _exmh_123`

**Dòng dữ liệu đích (Target Output):**
> `"win [MONEY] call [PHONE] visit [URL]",1`

### Ví dụ 2 — Email spam (Assassin/Enron dataset)
**Dữ liệu thô (Raw):**
> `Subject: Free Nokia! Body: Reply YES to 87077. begin pgp signature iQEVAwUBPw5A... end pgp signature ej8ijuqaqaqcaae... irish linux users group for unsubscription information list maintainer`

**Dòng dữ liệu đích (Target Output):**
> `"free nokia reply yes to [PHONE]",1`

*(Lưu ý: khối PGP, chuỗi base64 và footer mailing list đều bị xóa hoàn toàn trước khi xuất)*

---

## 4. Kiểm soát chất lượng (QA)
* **Tính mô-đun:** File `preprocess_fixed.py` phải chứa một hàm chính (ví dụ: `clean_text(raw_input)`) trả về đúng một chuỗi string duy nhất đã được xử lý.
* **Thử nghiệm chéo:** Chạy thử hàm này trên cả dòng dữ liệu SMS (Kaggle) và Email (Crawl). Nếu cả hai đều cho ra cùng một kiểu định dạng text sạch, khi đó script mới đạt yêu cầu.

Sau khi xong thì AI Model sẽ test trên dữ liệu mới
