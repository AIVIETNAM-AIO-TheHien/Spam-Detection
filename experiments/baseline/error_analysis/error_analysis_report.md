# Baseline Error Analysis

## 1. Mục tiêu

Phân tích các mẫu dự đoán sai của baseline TF-IDF + Multinomial Naive Bayes trên tập test cố định của dataset Kaggle SMS Spam Collection.

Phạm vi báo cáo này chỉ dựa trên tập test tiếng Anh hiện tại. Vì vậy, việc thiếu dữ liệu tiếng Việt không được xem là nguyên nhân trực tiếp gây lỗi trong phần phân tích này.
## 2. Tổng quan metrics

- accuracy: 0.9641
- macro_f1: 0.9125
- weighted_f1: 0.9617
- spam_precision: 1.0000
- spam_recall: 0.7321
- spam_f1: 0.8454

Nhận xét nhanh: accuracy cao nhưng spam recall thấp hơn rõ so với spam precision. Điều này cho thấy model dự đoán spam rất thận trọng: khi đã dự đoán là spam thì đúng, nhưng vẫn bỏ sót một phần spam thật.

## 3. Confusion matrix

| Actual label | Predicted ham | Predicted spam |
|---|---:|---:|
| ham | 724 | 0 |
| spam | 30 | 82 |

Vấn đề chính của baseline là False Negative, tức là spam bị dự đoán nhầm thành ham.

## 4. Tổng quan lỗi

- Tổng số mẫu test: 836
- Tổng số mẫu sai: 30
- Error rate: 3.59%
- False Positive: 0
- False Negative: 30

Tất cả lỗi trong tập test đều là spam bị phân loại nhầm thành ham. Không có trường hợp ham bị báo nhầm là spam.

## 5. Thống kê đặc điểm của lỗi

- 30/30 lỗi là False Negative.
- Độ dài trung bình: 121.23 ký tự.
- Số từ trung bình: 21.70 từ.
- 24/30 mẫu có chữ số, thường là số điện thoại, mã dịch vụ, giá tiền hoặc đầu số SMS.
- 18/30 mẫu có keyword spam trong bộ keyword đơn giản như `free`, `win`, `cash`, `prize`, `call`, `txt`, `stop`, `reply`.
- 12/30 mẫu không có keyword spam rõ ràng theo bộ keyword trên.
- 4/30 mẫu có URL.
- 3/30 mẫu có từ 5 từ trở xuống.
- 7/30 mẫu có từ 12 từ trở xuống.

## 6. Ví dụ False Positive

Không có False Positive trong tập test.

## 7. Ví dụ False Negative

- True: spam | Pred: ham
  Message: REMINDER FROM O2: To get 2.50 pounds free call credit and details of great offers pls reply 2 this text with your valid name, house no and postcode
- True: spam | Pred: ham
  Message: Orange brings you ringtones from all time Chart Heroes, with a free hit each week! Go to Ringtones & Pics on wap. To stop receiving these tips reply STOP.
- True: spam | Pred: ham
  Message: CLAIRE here am havin borin time & am now alone U wanna cum over 2nite? Chat now 09099725823 hope 2 C U Luv CLAIRE xx Calls£1/minmoremobsEMSPOBox45PO139WA
- True: spam | Pred: ham
  Message: Bloomberg -Message center +447797706009 Why wait? Apply for your future http://careers. bloomberg.com
- True: spam | Pred: ham
  Message: Want explicit SEX in 30 secs? Ring 02073162414 now! Costs 20p/min

## 8. Nhóm lỗi chính

- Spam có dấu hiệu thương mại nhưng bị xem là ham: ringtone, mobile service, credit, download.
- Spam giả dạng tin nhắn cá nhân hoặc hội thoại: `call`, `chat`, `talk`, `I`, `you`.
- Spam adult/dating service có số điện thoại, giá cước, `txt`, `stop` nhưng vẫn bị bỏ sót.
- Spam dạng thông báo/dịch vụ giống nội dung hợp lệ từ công ty hoặc customer service.
- Mẫu nhiễu, quá ngắn hoặc format bất thường làm model thiếu ngữ cảnh.

## 9. Kết luận

Baseline TF-IDF + Multinomial Naive Bayes đạt accuracy cao, nhưng lỗi tập trung hoàn toàn vào False Negative. Model không gây phiền cho người dùng bằng cách báo nhầm ham thành spam, nhưng lại bỏ sót 30/112 mẫu spam trong tập test.

Nguyên nhân chính quan sát được từ `prediction_errors.csv`: nhiều spam có văn phong giống tin nhắn cá nhân hoặc thông báo hợp lệ; nhiều mẫu có số điện thoại, mã dịch vụ hoặc giá cước nhưng baseline chưa tận dụng đủ mạnh các tín hiệu này; một số mẫu ngắn hoặc format bất thường khiến model thiếu ngữ cảnh.

## 10. Hướng cải thiện

- Thử Logistic Regression hoặc Linear SVM với TF-IDF.
- Bổ sung feature thủ công cho tín hiệu mạnh: số điện thoại, giá tiền, đầu số SMS, URL, `txt`, `stop`, `reply`, `call`.
- Chuẩn hóa số điện thoại, URL, ký hiệu tiền tệ và lỗi encoding trong preprocessing.
- Phân tích riêng các nhóm spam bị bỏ sót như mobile service, ringtone, dating/adult service và delivery/customer-service scam.
- Nếu muốn mở rộng sang bối cảnh Việt Nam, cần tạo tập đánh giá tiếng Việt riêng; đây là bài toán generalization ngoài dataset hiện tại.