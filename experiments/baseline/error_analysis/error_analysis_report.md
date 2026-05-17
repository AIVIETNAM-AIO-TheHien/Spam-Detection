# Baseline Error Analysis

## 1. Mục tiêu

Phân tích các mẫu dự đoán sai của baseline TF-IDF + Multinomial Naive Bayes trên tập test cố định của SMS Spam Collection.

Báo cáo này dùng cùng cấu hình preprocessing với bước evaluate baseline. Tin nhắn gốc và cột `preprocessed_text` đều được lưu trong các file CSV đầu ra.

## 2. Metrics

- accuracy: 0.9629
- macro_f1: 0.9092
- weighted_f1: 0.9603
- spam_precision: 1.0000
- spam_recall: 0.7232
- spam_f1: 0.8394

## 3. Confusion Matrix

| Nhãn thật | Dự đoán ham | Dự đoán spam |
|---|---:|---:|
| ham | 724 | 0 |
| spam | 31 | 81 |

## 4. Tổng quan lỗi

- Tổng số mẫu test: 836
- Tổng số mẫu sai: 31
- Error rate: 3.71%
- False Positive: 0
- False Negative: 31

Tất cả lỗi đều là False Negative: spam bị dự đoán nhầm thành ham.

## 5. Đặc điểm của lỗi

- Tỷ lệ False Negative trong lỗi: 31/31
- Độ dài trung bình: 124.55 ký tự
- Số từ trung bình: 22.52 từ
- Mẫu có chữ số: 24/31
- Mẫu có keyword spam: 19/31
- Mẫu không có keyword spam: 12/31
- Mẫu có URL: 3/31
- Mẫu có <= 5 từ: 2/31
- Mẫu có <= 12 từ: 6/31

## 6. Ví dụ False Positive

Không có False Positive trong tập test.

## 7. Ví dụ False Negative

- True: spam | Pred: ham
  Tin nhắn gốc: REMINDER FROM O2: To get 2.50 pounds free call credit and details of great offers pls reply 2 this text with your valid name, house no and postcode
  Sau preprocess: reminder from o2 to get 250 pounds free call credit and details of great offers pls reply 2 this text with your valid name house no and postcode
- True: spam | Pred: ham
  Tin nhắn gốc: This msg is for your mobile content order It has been resent as previous attempt failed due to network error Queries to customersqueries@netvision.uk.com
  Sau preprocess: this msg is for your mobile content order it has been resent as previous attempt failed due to network error queries to [email]
- True: spam | Pred: ham
  Tin nhắn gốc: Orange brings you ringtones from all time Chart Heroes, with a free hit each week! Go to Ringtones & Pics on wap. To stop receiving these tips reply STOP.
  Sau preprocess: orange brings you ringtones from all time chart heroes with a free hit each week go to ringtones pics on wap to stop receiving these tips reply stop
- True: spam | Pred: ham
  Tin nhắn gốc: CLAIRE here am havin borin time & am now alone U wanna cum over 2nite? Chat now 09099725823 hope 2 C U Luv CLAIRE xx Calls£1/minmoremobsEMSPOBox45PO139WA
  Sau preprocess: claire here am havin borin time am now alone u wanna cum over 2nite chat now [phone] hope 2 c u luv claire xx calls [money] minmoremobsemspobox45po139wa
- True: spam | Pred: ham
  Tin nhắn gốc: Bloomberg -Message center +447797706009 Why wait? Apply for your future http://careers. bloomberg.com
  Sau preprocess: bloomberg message center 447797706009 why wait apply for your future [url] bloombergcom

## 8. Nhận xét

- Model bỏ sót 31/112 mẫu spam trong tập test.
- Nên so sánh tin nhắn gốc với `preprocessed_text` trong `prediction_errors.csv` để kiểm tra preprocessing có làm mất tín hiệu spam hữu ích hay không.
- Các tín hiệu cần xem kỹ gồm dấu câu, ký hiệu tiền tệ, số điện thoại, URL, mã dịch vụ và các keyword như `free`, `claim`, `call`, `txt`, `stop`, `reply`.

## 9. Hướng tiếp theo

- So sánh báo cáo này với error analysis của baseline dùng raw text.
- Chạy ablation cho preprocessing, đặc biệt là `remove_punct: false`.
- Thử Logistic Regression hoặc Linear SVM với cùng bộ đặc trưng TF-IDF.
- Nếu recall vẫn thấp, bổ sung feature thủ công cho URL, phone, money, digit và spam keyword.