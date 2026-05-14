# Baseline Error Analysis

## 1. Mục tiêu

Phân tích các mẫu dự đoán sai của baseline TF-IDF + Multinomial Naive Bayes trên tập test cố định của dataset Kaggle SMS Spam Collection.

## 2. Tổng quan metrics

- Accuracy: 0.9641
- Macro F1: 0.9125
- Weighted F1: 0.9617
- Spam precision: 1.0000
- Spam recall: 0.7321
- Spam F1: 0.8454

Nhận xét nhanh: accuracy cao nhưng spam recall thấp hơn rõ so với spam precision. Điều này cho thấy model dự đoán spam rất thận trọng: khi đã dự đoán là spam thì đúng, nhưng vẫn bỏ sót một phần spam thật.

## 3. Confusion matrix

| Actual label | Predicted ham | Predicted spam |
|---|---:|---:|
| ham | 724 | 0 |
| spam | 30 | 82 |

Từ confusion matrix:

- True Negative: 724 tin ham được dự đoán đúng là ham.
- False Positive: 0 tin ham bị dự đoán nhầm thành spam.
- False Negative: 30 tin spam bị dự đoán nhầm thành ham.
- True Positive: 82 tin spam được dự đoán đúng là spam.

Vấn đề chính của baseline là False Negative, tức là bỏ sót spam.

## 4. Tổng quan lỗi

- Tổng số mẫu test: 836
- Tổng số mẫu sai: 30
- Error rate: 3.59%
- False Positive: 0
- False Negative: 30

Tất cả lỗi trong tập test đều là spam bị phân loại nhầm thành ham. Không có trường hợp ham bị báo nhầm là spam.

## 5. Thống kê đặc điểm của 30 lỗi

Từ file `prediction_errors.csv`:

- 30/30 lỗi là False Negative.
- Độ dài trung bình: 121.23 ký tự.
- Số từ trung bình: 21.70 từ.
- 24/30 mẫu có chữ số, thường là số điện thoại, mã dịch vụ, giá tiền hoặc đầu số SMS.
- 18/30 mẫu có keyword spam trong bộ keyword đơn giản như `free`, `win`, `cash`, `prize`, `call`, `txt`, `stop`, `reply`.
- 12/30 mẫu không có keyword spam rõ ràng theo bộ keyword trên.
- 4/30 mẫu có URL.
- 3/30 mẫu rất ngắn, từ 5 từ trở xuống.
- 7/30 mẫu có từ 12 từ trở xuống.

Điểm đáng chú ý là nhiều mẫu bị bỏ sót vẫn chứa tín hiệu spam khá rõ như số điện thoại, đầu số dịch vụ, giá cước, từ khóa `free`, `call`, `txt`, `stop`. Điều này cho thấy model không chỉ sai vì thiếu keyword, mà có thể vì trọng số học được từ TF-IDF + Naive Bayes chưa đủ mạnh cho các pattern này trong một số ngữ cảnh.

## 6. Nhóm lỗi chính

### 6.1. Spam có dấu hiệu thương mại nhưng bị xem là ham

Ví dụ:

- `REMINDER FROM O2: To get 2.50 pounds free call credit...`
- `Orange brings you ringtones... with a free hit each week!`
- `Your weekly Cool-Mob tones are ready to download...`

Nhóm này chứa các nội dung quảng cáo, ringtone, dịch vụ mobile, credit hoặc download. Dù có một số từ khóa spam, model vẫn dự đoán là ham. Khả năng là các token trong nhóm này không đủ phân biệt trong không gian TF-IDF, hoặc nội dung có nhiều từ trung tính làm giảm ảnh hưởng của keyword spam.

### 6.2. Spam giả dạng tin nhắn cá nhân hoặc hội thoại

Ví dụ:

- `Sorry I missed your call let's talk when you have the time...`
- `Can U get 2 phone NOW? I wanna chat...`
- `I'd like to tell you my deepest darkest fantasies...`

Các mẫu này dùng văn phong giống hội thoại cá nhân: `I`, `you`, `call`, `chat`, `talk`, `time`. Đây là kiểu nội dung dễ bị Naive Bayes nhầm với ham vì dataset ham cũng có nhiều tin nhắn giao tiếp đời thường.

### 6.3. Spam người lớn hoặc dating service

Ví dụ:

- `Want explicit SEX in 30 secs? Ring...`
- `Talk sexy!! Make new friends or fall in love...`
- `Mila, age23, blonde, new in UK...`

Nhóm này có nhiều tín hiệu spam như số điện thoại, giá cước, `txt`, `stop`, nhưng vẫn bị dự đoán ham. Điều này cho thấy baseline có thể chưa học đủ tốt các pattern spam dạng adult/dating service, hoặc các từ khóa này xuất hiện rải rác và bị nhiễu bởi từ ngữ hội thoại.

### 6.4. Spam dạng thông báo/dịch vụ

Ví dụ:

- `Customer service announcement. We recently tried to make a delivery...`
- `Bloomberg -Message center... Apply for your future...`
- `You are now unsubscribed all services...`

Các mẫu này có giọng điệu giống thông báo hợp lệ từ dịch vụ hoặc công ty. Với mô hình bag-of-words như TF-IDF + Naive Bayes, model không hiểu được ý định lừa đảo/quảng cáo phía sau câu, nên dễ phân loại thành ham.

### 6.5. Mẫu nhiễu, quá ngắn hoặc bất thường

Ví dụ:

- `#ERROR!`
- `FROM 88066 LOST £12 HELP`
- URL hoặc text bị lỗi encoding như `http//tms. widelive.com/...`

Các mẫu này có ít ngữ cảnh hoặc format bất thường. Với số từ ít, model có ít token để dựa vào, nên dễ bỏ sót spam.

## 7. Kết luận

Baseline TF-IDF + Multinomial Naive Bayes đạt accuracy cao, nhưng lỗi tập trung hoàn toàn vào False Negative. Model không gây phiền cho người dùng bằng cách báo nhầm ham thành spam, nhưng lại bỏ sót 30/112 mẫu spam trong tập test.

Nguyên nhân chính quan sát được từ `prediction_errors.csv`:

- Nhiều spam có văn phong giống tin nhắn cá nhân hoặc thông báo hợp lệ.
- Một số spam thuộc domain mobile service, ringtone, dating/adult service, delivery/customer service.
- Nhiều mẫu có số điện thoại, mã dịch vụ hoặc giá cước, nhưng baseline chưa tận dụng đủ mạnh các tín hiệu này.
- Một số mẫu ngắn hoặc format bất thường khiến model thiếu ngữ cảnh.
- 12/30 lỗi không chứa keyword spam rõ ràng theo bộ keyword đơn giản, nên keyword matching đơn thuần không đủ.

## 8. Hướng cải thiện

- Thử Logistic Regression hoặc Linear SVM với TF-IDF, vì hai mô hình này thường xử lý bài toán text classification tốt hơn Multinomial Naive Bayes trong nhiều trường hợp.
- Bổ sung feature thủ công cho các tín hiệu mạnh: có số điện thoại, có giá tiền, có đầu số SMS, có URL, có từ `txt`, `stop`, `reply`, `call`.
- Kiểm tra lại preprocessing để chuẩn hóa số điện thoại, URL, ký hiệu tiền tệ và lỗi encoding.
- Phân tích riêng các nhóm spam bị bỏ sót như mobile service, ringtone, dating/adult service và delivery/customer-service scam.

