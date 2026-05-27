# So sánh Error Analysis Trước và Sau Preprocess
> Note: Báo cáo này thuộc baseline experiment cũ và có thể không còn khớp hoàn toàn với fixed split hiện tại. Kết quả model chính nên tham khảo các report mới trong `docs/` và các metrics tương ứng trong `experiments/logistic_regression/` hoặc `experiments/linear_svc/`.

## 1. Mục tiêu

Báo cáo này so sánh các mẫu dự đoán sai của baseline TF-IDF + Multinomial Naive Bayes trước và sau khi tích hợp preprocessing.

Nguồn so sánh:

- Trước preprocess: `prediction_errors_notusepreprocess.csv`
- Sau preprocess: `experiments/baseline/error_analysis/prediction_errors.csv`

## 2. Tổng quan thay đổi

| Phiên bản | Tổng lỗi | False Positive | False Negative |
|---|---:|---:|---:|
| Chưa dùng preprocess | 30 | 0 | 30 |
| Đã dùng preprocess | 31 | 0 | 31 |

Sau preprocess, tổng số lỗi tăng nhẹ từ 30 lên 31. Tất cả lỗi ở cả hai phiên bản đều là False Negative, tức là spam bị dự đoán nhầm thành ham.

Điểm quan trọng là phần lớn lỗi không thay đổi:

- 29 mẫu lỗi xuất hiện ở cả hai phiên bản.
- 2 mẫu trở thành lỗi mới sau preprocess.
- 1 mẫu lỗi cũ được model dự đoán đúng sau preprocess.

## 3. So sánh đặc điểm lỗi

| Đặc điểm | Chưa preprocess | Đã preprocess |
|---|---:|---:|
| Số lỗi | 30 | 31 |
| Độ dài trung bình | 121.23 ký tự | 124.55 ký tự |
| Số từ trung bình | 21.70 từ | 22.52 từ |
| Có URL | 4 | 3 |
| Có chữ số | 24 | 24 |
| Có spam keyword | 18 | 19 |
| Có <= 5 từ | 3 | 2 |
| Có <= 12 từ | 7 | 6 |

Nhìn tổng thể, preprocess không làm thay đổi hoàn toàn nhóm lỗi. Các lỗi vẫn tập trung vào spam có số điện thoại, mã dịch vụ, tiền, keyword quảng cáo hoặc nội dung giống thông báo dịch vụ.

## 4. Các lỗi giữ nguyên

Có 29 mẫu vẫn bị bỏ sót ở cả hai phiên bản. Điều này cho thấy nguyên nhân chính không chỉ nằm ở preprocessing, mà còn ở giới hạn của baseline TF-IDF + Multinomial Naive Bayes.

Các nhóm lỗi ổn định gồm:

- Spam thương mại hoặc dịch vụ mobile: ringtone, credit, download, content order.
- Spam có văn phong giống tin nhắn cá nhân hoặc thông báo hợp lệ.
- Spam có số điện thoại, mã dịch vụ, giá tiền nhưng vẫn bị model xem là ham.
- Mẫu có nhiều từ trung tính làm yếu tín hiệu spam trong biểu diễn bag-of-words.

## 5. Lỗi mới sau preprocess

Sau preprocess có 2 mẫu mới bị bỏ sót.

### Mẫu 4060

Tin nhắn gốc:

```text
Moby Pub Quiz.Win a £100 High Street prize if u know who the new Duchess of Cornwall will be? Txt her first name to 82277.unsub STOP £1.50 008704050406 SP
```

Sau preprocess:

```text
moby pub quizwin a [money] high street prize if u know who the new duchess of cornwall will be txt her first name to 82277unsub stop [money] [phone] 06 sp
```

Khả năng nguyên nhân:

- Dấu câu bị xóa làm một số token bị dính lại, ví dụ `Quiz.Win` thành `quizwin` và `82277.unsub` thành `82277unsub`.
- Ký hiệu tiền và số điện thoại được chuẩn hóa thành `[money]`, `[phone]`, giúp tổng quát hóa nhưng cũng làm mất chi tiết cụ thể như `£100`, `£1.50`, `008704050406`.
- Tin nhắn vẫn có keyword spam như `prize`, `txt`, `stop`, nhưng các token bị dính và việc thay thế entity có thể làm vector TF-IDF khác so với bản raw.

### Mẫu 5102

Tin nhắn gốc:

```text
This msg is for your mobile content order It has been resent as previous attempt failed due to network error Queries to customersqueries@netvision.uk.com
```

Sau preprocess:

```text
this msg is for your mobile content order it has been resent as previous attempt failed due to network error queries to [email]
```

Khả năng nguyên nhân:

- Email `customersqueries@netvision.uk.com` được thay bằng token chung `[email]`.
- Việc chuẩn hóa email làm mất các token domain như `netvision`, `uk`, `com`; nếu các token này từng đóng góp tín hiệu spam trong bản raw thì model sau preprocess sẽ yếu hơn ở mẫu này.
- Nội dung còn lại giống thông báo dịch vụ hợp lệ: `mobile content order`, `network error`, `queries`. Với Naive Bayes, những token này có thể không đủ mạnh để kéo mẫu về spam.

## 6. Mẫu được sửa sau preprocess

Có 1 mẫu từng bị bỏ sót trước preprocess nhưng được dự đoán đúng sau preprocess.

### Mẫu 5466

Tin nhắn gốc:

```text
http//tms. widelive.com/index. wml?id=820554ad0a1705572711&first=true¡C C Ringtone¡
```

Sau preprocess:

```text
httptms widelivecomindex wmlid820554ad0a1705572711firsttrue¡c c ringtone¡
```

Khả năng nguyên nhân:

- Preprocess xóa dấu câu và lowercase, giúp gom lại các mảnh URL bị format lỗi.
- Token `ringtone` vẫn được giữ lại, đây là tín hiệu spam khá rõ trong dataset.
- Dù URL không được chuẩn hóa thành `[url]` vì format gốc là `http//` thay vì `http://`, việc làm sạch vẫn giúp biểu diễn nhất quán hơn đủ để model dự đoán đúng.

## 7. Kết luận

Preprocess làm thay đổi nhẹ hành vi của baseline:

- Dev recall tăng nhưng test recall giảm nhẹ.
- Trên test, số False Negative tăng từ 30 lên 31.
- Phần lớn lỗi vẫn giữ nguyên, nên nguyên nhân chính vẫn là giới hạn của mô hình baseline hơn là lỗi preprocess đơn thuần.
- Một số bước preprocess có thể làm mất tín hiệu cụ thể, nhất là xóa dấu câu và chuẩn hóa email/phone/money thành token chung.

Nguyên nhân hợp lý nhất cho việc test recall giảm là preprocess vừa giúp tổng quát hóa, vừa làm mất một số tín hiệu thưa nhưng hữu ích trong SMS spam. Với dataset nhỏ, chỉ 1-2 mẫu thay đổi cũng đủ làm recall test dao động.

## 8. Hướng tiếp theo

- Chạy ablation với `remove_punct: false` để kiểm tra tác động của việc xóa dấu câu.
- Thử giữ entity token nhưng không làm dính token quanh dấu câu, đặc biệt các pattern như `Quiz.Win` hoặc `82277.unsub`.
- So sánh thêm cấu hình `normalize_entities_flag: false` để xem việc thay email/phone/money bằng token chung có làm giảm tín hiệu cụ thể hay không.
- Thử Logistic Regression hoặc Linear SVM với cùng TF-IDF, vì hai mô hình này thường xử lý text classification tốt hơn Multinomial Naive Bayes.
- Nếu vẫn ưu tiên recall spam, cân nhắc thêm feature thủ công cho `has_phone`, `has_money`, `has_url`, `has_digit`, `has_spam_keyword`.
