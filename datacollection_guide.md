# Hướng dẫn thu thập dữ liệu spam tiếng Anh

## Mục đích
Bổ sung các domain spam tiếng Anh còn thiếu trong dataset hiện tại, bao gồm:
- **Phishing** (lừa đảo ngân hàng, PayPal, xác minh tài khoản)
- **Lottery / Prize scams** (trúng thưởng, xổ số)
- **Product spam** (quảng cáo thuốc, tình dục, chứng khoán)
- **MLM / Get rich quick** (đa cấp, cơ hội làm giàu)

Đồng thời thu thập thêm email ham (hợp pháp) bằng tiếng Anh để cân bằng dữ liệu.

## Các nguồn dữ liệu công khai

| Loại spam | Nguồn | Phương pháp thu thập | Số lượng mục tiêu |
|-----------|-------|----------------------|-------------------|
| **Phishing** | PhishTank, OpenPhish, Kaggle "Phishing Email Dataset" | Tải CSV/JSON có sẵn (API của PhishTank) | 500+ |
| **Lottery / Prize** | SpamAssassin corpus (đã có), 419scam.org | Tải trực tiếp hoặc copy tay từ diễn đàn | 300 |
| **Product spam** | Spam archive (spamarchive.org), Reddit r/spam | Crawl bằng Python (requests, BeautifulSoup) | 400 |
| **MLM / Get rich** | Reddit (r/antiMLM, r/scams), các group Facebook công khai | Dùng Reddit API (Pushshift) hoặc copy tay | 300 |
| **Ham (legit email)** | Enron corpus, Apache mailing lists (e.g., SpamAssassin-talk) | Tải từ các kho dữ liệu mở (UCI, Kaggle) | 1000+ |

## Hướng dẫn chi tiết

### 1. Phishing
- Truy cập [PhishTank](https://www.phishtank.com/developer_info.php), đăng ký API key.
- Tải danh sách phishing mới nhất dạng JSON: `https://data.phishtank.com/data/online-valid.json`
- Mỗi mục chứa URL, tiêu đề, mô tả – có thể dùng làm nội dung spam.

### 2. Lottery / Prize scams
- Tải bộ [SpamAssassin public corpus](https://spamassassin.apache.org/old/publiccorus/) (đã có trong dự án).
- Tìm kiếm từ khóa "lottery", "winner", "prize", "inheritance" trong các file spam.

### 3. Product spam
- Truy cập [Spam Archive](http://www.spamarchive.org/) – đã được gắn nhãn.
- Sử dụng `wget` hoặc Python `requests` để tải các file text.

### 4. MLM / Get rich quick
- Dùng Reddit API (Pushshift) để query các bài viết trong `r/antiMLM` hoặc `r/scams`.
- Ví dụ query: `https://api.pushshift.io/reddit/search/submission?subreddit=antiMLM&size=500`
- Lọc các bài có chứa email hoặc nội dung spam.

### 5. Ham (legitimate email)
- Tải [Enron Email Dataset](https://www.cs.cmu.edu/~enron/) (khoảng 500k email).
- Chọn ngẫu nhiên 1000 email từ thư mục `enron1/ham/` (hoặc dùng bộ đã được xử lý trên Kaggle).

## Định dạng lưu trữ

Lưu dữ liệu thu thập được dưới dạng CSV hoặc JSON với ít nhất hai cột:
- `text`: nội dung email/tin nhắn thô (chưa xử lý)
- `label`: `spam` hoặc `ham`
Có thể thêm cột `source` để ghi nhận nguồn.

Ví dụ (JSON):
```json
{"text": "You have won $1,000,000! Click here to claim your prize now.", "label": "spam", "source": "phishTank_2025-05-14"}