# Báo cáo so sánh Logistic Regression và LinearSVC

Ngày báo cáo: 2026-05-20

## 1. Mục tiêu

Báo cáo này tổng hợp các lần chạy của hai nhóm mô hình:

- Logistic Regression
- LinearSVC

Mục tiêu là giải thích vì sao kết quả thay đổi giữa các cấu hình, mô hình nào đang tốt hơn, và nên chọn mô hình nào cho bước tiếp theo.

Tất cả thí nghiệm dùng cùng dataset và cùng split:

- Dataset: `data/custom/Assassin_preprocessed.csv`
- Tổng số mẫu: 5178
- Train/dev/test: 3624 / 777 / 777
- Test set: 555 ham, 222 spam
- Feature chính: TF-IDF
- Preprocess: `enabled: true`
- Split seed: 42

## 2. Tóm tắt kết quả chính

| Model | Run tốt nhất theo quy trình dev | Test spam precision | Test spam recall | Test spam F1 | Tổng lỗi | False positive | False negative |
|---|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | `lr_c1_0_ng12_min2` | 0.9595 | 0.9595 | 0.9595 | 18 | 9 | 9 |
| LinearSVC | `svc_c1_0_ng12_min2` | 0.9478 | 0.9820 | 0.9646 | 16 | 12 | 4 |

LinearSVC có spam F1 cao hơn Logistic Regression, đồng thời giảm false negative từ 9 xuống 4. Với bài toán spam detection, false negative là lỗi quan trọng vì đó là spam bị bỏ sót.

## 3. Kết quả Logistic Regression

| Run ID | C | N-gram | Min DF | Threshold | Test spam precision | Test spam recall | Test spam F1 | Tổng lỗi | FP | FN |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lr_c1_0_ng12_min2` | 1.0 | [1, 2] | 2 | 0.48 | 0.9595 | 0.9595 | 0.9595 | 18 | 9 | 9 |
| `lr_c2_0_ng12_min2` | 2.0 | [1, 2] | 2 | 0.45 | 0.9509 | 0.9595 | 0.9552 | 20 | 11 | 9 |
| `lr_c1_0_ng13_min2` | 1.0 | [1, 3] | 2 | 0.46 | 0.9427 | 0.9640 | 0.9532 | 21 | 13 | 8 |
| `lr_c0_5_ng12_min2` | 0.5 | [1, 2] | 2 | 0.51 | 0.9545 | 0.9459 | 0.9502 | 22 | 10 | 12 |
| `lr_c1_0_ng12_min1` | 1.0 | [1, 2] | 1 | 0.46 | 0.9383 | 0.9595 | 0.9488 | 23 | 14 | 9 |

Nhận xét:

- `C=1.0`, `ngram_range=[1,2]`, `min_df=2` là cấu hình cân bằng tốt nhất.
- Tăng `C` lên 2.0 làm model bớt regularize hơn, nhưng không cải thiện F1. False positive tăng từ 9 lên 11.
- Dùng n-gram `[1,3]` giúp recall tăng nhẹ, nhưng precision giảm nhiều hơn, nên F1 giảm.
- `min_df=1` giữ lại nhiều token hiếm hơn, nhưng không giúp model tốt hơn. Khả năng cao các token hiếm làm tăng nhiễu.

Kết luận cho Logistic Regression: mô hình đã rất tốt, nhưng vẫn còn 9 spam bị bỏ sót trên test set.

## 4. Kết quả LinearSVC

LinearSVC có hai nhóm kết quả cần phân biệt:

- `default_test_*`: dùng threshold mặc định của LinearSVC, tức decision boundary tại `0`.
- `test_*`: dùng threshold được chọn trên dev set theo tiêu chí `best_dev_spam_f1`.

Khi báo cáo nghiêm túc, nên ưu tiên kết quả `test_*`, vì threshold được chọn trên dev set, không chọn theo test set.

| Run ID | C | N-gram | Min DF | Threshold chọn trên dev | Test spam precision | Test spam recall | Test spam F1 | Tổng lỗi | FP | FN |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `svc_c1_0_ng12_min2` | 1.0 | [1, 2] | 2 | -0.2165 | 0.9478 | 0.9820 | 0.9646 | 16 | 12 | 4 |
| `svc_c1_0_ng13_min2` | 1.0 | [1, 3] | 2 | -0.1954 | 0.9437 | 0.9820 | 0.9625 | 17 | 13 | 4 |
| `svc_c1_0_ng12_min1` | 1.0 | [1, 2] | 1 | -0.2345 | 0.9437 | 0.9820 | 0.9625 | 17 | 13 | 4 |
| `svc_c2_0_ng12_min2` | 2.0 | [1, 2] | 2 | -0.2172 | 0.9476 | 0.9775 | 0.9623 | 17 | 12 | 5 |
| `svc_c0_5_ng12_min2` | 0.5 | [1, 2] | 2 | -0.2080 | 0.9435 | 0.9775 | 0.9602 | 18 | 13 | 5 |

Nhận xét:

- LinearSVC nhìn chung bắt spam tốt hơn Logistic Regression, thể hiện ở recall rất cao.
- Best run `svc_c1_0_ng12_min2` chỉ bỏ sót 4 spam, ít nhất trong các mô hình đã thử.
- Khi dùng n-gram `[1,3]`, recall giữ rất cao nhưng false positive tăng, nên F1 không vượt run `[1,2]`.
- `min_df=1` cũng không giúp SVM tốt hơn, tương tự Logistic Regression.
- `C=1.0` tiếp tục là điểm cân bằng tốt nhất. `C=2.0` không cải thiện, còn `C=0.5` hơi yếu hơn.

## 5. Vì sao kết quả thay đổi?

### 5.1. Ảnh hưởng của `C`

`C` điều khiển mức regularization:

- `C` thấp hơn: model bị ràng buộc mạnh hơn, ít overfit hơn nhưng có thể bỏ sót tín hiệu spam.
- `C` cao hơn: model linh hoạt hơn, học mạnh hơn theo feature, nhưng dễ tăng false positive.

Trong cả Logistic Regression và LinearSVC, `C=1.0` là điểm cân bằng tốt nhất. `C=0.5` hơi bảo thủ hơn, còn `C=2.0` không đem lại lợi ích rõ ràng.

### 5.2. Ảnh hưởng của n-gram

`ngram_range=[1,2]` dùng unigram và bigram. Đây là cấu hình khá cân bằng: đủ bắt các cụm như `free trial`, `click here`, `remove reply`, nhưng không làm feature space quá nhiễu.

`ngram_range=[1,3]` thêm trigram. Cấu hình này có thể bắt thêm pattern dài hơn, nên recall đôi khi tăng. Tuy nhiên, nó cũng thêm nhiều feature hiếm, làm precision giảm. Vì vậy F1 tổng thể không tốt hơn.

### 5.3. Ảnh hưởng của `min_df`

`min_df=1` giữ cả token xuất hiện rất ít. Với spam/email, token hiếm có thể là:

- mã tracking
- chuỗi random
- tên riêng
- ký tự lỗi
- token chỉ xuất hiện trong một vài email

Những token này không nhất thiết giúp phân biệt spam tốt hơn. Kết quả cho thấy `min_df=2` ổn hơn vì loại bớt nhiễu.

### 5.4. Ảnh hưởng của threshold

Logistic Regression có xác suất `P(spam)`, nên threshold kiểu `0.48` khá dễ hiểu: nếu xác suất spam vượt ngưỡng thì dự đoán spam.

LinearSVC không có xác suất trực tiếp. Nó có `decision_function`, tức là điểm khoảng cách tới hyperplane:

- score cao hơn: nghiêng về spam hơn
- score thấp hơn: nghiêng về ham hơn
- threshold mặc định thường là `0`

Khi chọn threshold âm như `-0.2165`, model trở nên nhạy hơn với spam. Kết quả là recall tăng mạnh, false negative giảm, nhưng false positive tăng.

Với spam detection, trade-off này hợp lý vì bỏ sót spam thường tệ hơn cảnh báo nhầm một ít ham.

## 6. Vì sao LinearSVC tốt hơn Logistic Regression?

Cả hai đều là linear model trên TF-IDF, nhưng cách tối ưu khác nhau:

- Logistic Regression tối ưu log-loss, học xác suất phân lớp.
- LinearSVC tối ưu hinge loss, tập trung tạo margin phân tách giữa hai lớp.

Với dữ liệu text TF-IDF nhiều chiều, LinearSVC thường rất mạnh vì margin-based learning hợp với bài toán sparse high-dimensional như email/spam classification.

Trong kết quả hiện tại:

- Logistic Regression bỏ sót 9 spam.
- LinearSVC tuned chỉ bỏ sót 4 spam.
- LinearSVC có spam F1 cao hơn.
- LinearSVC có recall cao hơn đáng kể.

Điểm đổi lại là LinearSVC có false positive nhiều hơn:

- Logistic Regression FP: 9
- LinearSVC FP: 12

Tức là SVM bắt thêm được 5 spam bị Logistic Regression bỏ sót, nhưng đánh nhầm thêm 3 ham thành spam. Với bài toán spam detection, đây là trade-off chấp nhận được nếu ưu tiên chặn spam.

## 7. Có nên chọn kết quả LinearSVC default cao nhất không?

Một số run LinearSVC dùng threshold mặc định có test spam F1 rất cao, ví dụ:

| Run ID | Default test spam precision | Default test spam recall | Default test spam F1 |
|---|---:|---:|---:|
| `svc_c1_0_ng13_min2` | 0.9773 | 0.9685 | 0.9729 |
| `svc_c0_5_ng12_min2` | 0.9729 | 0.9685 | 0.9707 |
| `svc_c1_0_ng12_min2` | 0.9772 | 0.9640 | 0.9705 |

Các kết quả này rất tốt, nhưng không nên lấy chúng làm kết luận chính nếu mục tiêu là đánh giá nghiêm túc. Lý do là model/threshold nên được chọn trên dev set, còn test set chỉ dùng để báo cáo cuối cùng.

Vì vậy, trong báo cáo chính nên chọn:

`svc_c1_0_ng12_min2` với threshold chọn trên dev.

Có thể ghi chú thêm rằng threshold mặc định của LinearSVC cũng cho kết quả test rất mạnh và nên được kiểm chứng thêm bằng cross-validation hoặc một test set mới.

## 8. Mô hình nên chọn

Nên chọn:

```text
LinearSVC: svc_c1_0_ng12_min2
C = 1.0
ngram_range = [1, 2]
min_df = 2
threshold = -0.2165
```

Lý do:

- Có spam F1 cao hơn Logistic Regression.
- Có spam recall cao nhất trong nhóm cấu hình tốt.
- False negative thấp nhất: chỉ 4 spam bị bỏ sót.
- Cấu hình đơn giản, không cần n-gram quá dài.
- Được chọn theo dev set, nên phương pháp đánh giá sạch hơn chọn theo test.

## 9. Kết luận

Logistic Regression đã là một cải thiện lớn so với baseline Naive Bayes, nhưng LinearSVC hiện là mô hình tốt hơn cho bài toán này.

Kết luận cuối:

| Model | Spam F1 | Spam recall | False negative | Nhận xét |
|---|---:|---:|---:|---|
| Naive Bayes baseline | 0.8840 | 0.8063 | 43 | Làm mốc baseline ổn, nhưng bỏ sót nhiều spam |
| Logistic Regression tốt nhất | 0.9595 | 0.9595 | 9 | Rất tốt, cân bằng precision/recall |
| LinearSVC tốt nhất theo dev | 0.9646 | 0.9820 | 4 | Nên chọn làm model chính hiện tại |

LinearSVC nên được chọn làm model ứng viên chính hiện tại. Bước tiếp theo nên kiểm chứng thêm bằng cross-validation hoặc test set mới, sau đó mới chốt production model.
