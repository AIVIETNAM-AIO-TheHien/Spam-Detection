# Báo cáo so sánh Baseline và Logistic Regression tốt nhất

Ngày báo cáo: 2026-05-19

## 1. Mục tiêu

Báo cáo này so sánh mô hình baseline TF-IDF + Multinomial Naive Bayes với mô hình Logistic Regression tốt nhất sau khi dataset `data/custom/Assassin_preprocessed.csv` được cập nhật lại.

Tất cả kết quả bên dưới dùng cùng một split mới, được tạo lại bằng seed 42:

- File split: `experiments/splits/split_indices.json`
- Tổng số mẫu: 5178
- Train/dev/test: 3624 / 777 / 777 mẫu
- Phân bố nhãn toàn bộ data: 3698 ham, 1480 spam
- Test set: 555 ham, 222 spam
- Preprocess: `enabled: true` cho cả baseline và Logistic Regression

## 2. Kiểm tra các lần train

Đã chạy lại toàn bộ pipeline:

- `scripts/create_split.py`
- `scripts/train_baseline.py`
- `scripts/evaluate_baseline.py`
- `scripts/train_logistic_regression.py`
- `scripts/evaluate_logistic_regression.py`
- `scripts/run_logistic_experiments.py`

Đã chạy 5 cấu hình Logistic Regression. Mỗi run đều có đầy đủ artifact phân tích:

- `metrics.json`
- `config_snapshot.json`
- `error_analysis/all_predictions.csv`
- `error_analysis/prediction_errors.csv`
- `error_analysis/confusion_matrix.csv`
- `error_analysis/error_summary.json`
- `error_analysis/error_analysis_report.md`

Bảng tổng hợp kết quả nằm ở `experiments/logistic_regression/experiment_summary.csv`.

## 3. Kết quả các run Logistic Regression

| Hạng | Run ID | C | N-gram | Min DF | Threshold | Test spam precision | Test spam recall | Test spam F1 |
|---:|---|---:|---|---:|---:|---:|---:|---:|
| 1 | `lr_c1_0_ng12_min2` | 1.0 | [1, 2] | 2 | 0.48 | 0.9595 | 0.9595 | 0.9595 |
| 2 | `lr_c2_0_ng12_min2` | 2.0 | [1, 2] | 2 | 0.45 | 0.9509 | 0.9595 | 0.9552 |
| 3 | `lr_c1_0_ng13_min2` | 1.0 | [1, 3] | 2 | 0.46 | 0.9427 | 0.9640 | 0.9532 |
| 4 | `lr_c0_5_ng12_min2` | 0.5 | [1, 2] | 2 | 0.51 | 0.9545 | 0.9459 | 0.9502 |
| 5 | `lr_c1_0_ng12_min1` | 1.0 | [1, 2] | 1 | 0.46 | 0.9383 | 0.9595 | 0.9488 |

Run tốt nhất theo `test_spam_f1` là `lr_c1_0_ng12_min2`.

## 4. Cấu hình model tốt nhất

Run: `experiments/logistic_regression/runs/lr_c1_0_ng12_min2`

```yaml
model:
  type: logistic_regression
  C: 1.0
  class_weight: balanced
  max_iter: 2000
  solver: liblinear

vectorizer:
  type: tfidf
  max_features: 50000
  ngram_range: [1, 2]
  min_df: 2
  max_df: 0.95
  lowercase: true
  strip_accents: unicode

threshold:
  selected_on_dev: 0.48
```

Threshold `0.48` được chọn trên dev set theo tiêu chí `best_dev_spam_f1`, sau đó mới đánh giá trên test set.

## 5. So sánh metric trên test set

| Metric | Baseline Naive Bayes | Logistic Regression tốt nhất | Chênh lệch |
|---|---:|---:|---:|
| Accuracy | 0.9395 | 0.9768 | +0.0373 |
| Macro F1 | 0.9215 | 0.9716 | +0.0501 |
| Weighted F1 | 0.9376 | 0.9768 | +0.0392 |
| Spam precision | 0.9781 | 0.9595 | -0.0187 |
| Spam recall | 0.8063 | 0.9595 | +0.1532 |
| Spam F1 | 0.8840 | 0.9595 | +0.0755 |

Kết quả quan trọng nhất là spam recall tăng từ `0.8063` lên `0.9595`. Điều này cho thấy Logistic Regression bỏ sót spam ít hơn baseline rất nhiều, trong khi precision chỉ giảm nhẹ.

## 6. Confusion matrix

Baseline Naive Bayes:

| Nhãn thật | Dự đoán ham | Dự đoán spam |
|---|---:|---:|
| Ham | 551 | 4 |
| Spam | 43 | 179 |

Logistic Regression tốt nhất:

| Nhãn thật | Dự đoán ham | Dự đoán spam |
|---|---:|---:|
| Ham | 546 | 9 |
| Spam | 9 | 213 |

## 7. Phân tích lỗi

Baseline có 47 lỗi trên 777 mẫu test:

- False positive: 4
- False negative: 43
- Tổng lỗi: 47

Logistic Regression tốt nhất có 18 lỗi trên 777 mẫu test:

- False positive: 9
- False negative: 9
- Tổng lỗi: 18

So với baseline, Logistic Regression:

- Giảm false negative từ 43 xuống 9, tức là bớt bỏ sót 34 mẫu spam.
- Tăng false positive từ 4 lên 9, tức là thêm 5 mẫu ham bị cảnh báo nhầm là spam.
- Giảm tổng lỗi từ 47 xuống 18.

Đây là trade-off tốt cho bài toán spam detection: model mới bắt được nhiều spam hơn đáng kể, trong khi số ham bị nhầm thành spam chỉ tăng nhẹ.

## 8. Nhận xét về các run

`lr_c1_0_ng12_min2` là cấu hình cân bằng tốt nhất hiện tại. Run `lr_c1_0_ng13_min2` có spam recall cao hơn một chút (`0.9640`) nhưng precision thấp hơn, nên spam F1 kém hơn. Run `lr_c2_0_ng12_min2` có kết quả gần sát run tốt nhất, nhưng không vượt được `C=1.0` trên dataset đã sửa.

`min_df=1` không cải thiện kết quả. Điều này cho thấy việc giữ lại quá nhiều token hiếm có thể làm model nhiễu hơn thay vì học được tín hiệu tốt hơn. Cấu hình `[1, 2]` với `min_df=2` đang là lựa chọn tốt nhất trong nhóm đã thử.

## 9. Kết luận

Logistic Regression tốt nhất hiện tại nên được chọn làm model ứng viên mới thay baseline:

- Spam F1 tăng từ `0.8840` lên `0.9595`.
- Spam recall tăng từ `0.8063` lên `0.9595`.
- Tổng lỗi trên test set giảm từ 47 xuống 18.
- False negative giảm mạnh từ 43 xuống 9.
- Model vẫn dùng TF-IDF nên dễ giải thích, dễ deploy và phù hợp với pipeline hiện tại.

Nên tiếp tục đọc `prediction_errors.csv` của run `lr_c1_0_ng12_min2`, tập trung vào 9 false negative còn lại và 9 false positive mới. Nếu mục tiêu ưu tiên chặn spam mạnh hơn nữa, có thể thử `lr_c1_0_ng13_min2` hoặc giảm threshold. Nếu mục tiêu ưu tiên giảm làm phiền user, có thể tăng threshold trên run tốt nhất.

## 10. Lưu ý về artifact

Dataset đã thay đổi từ 5747 mẫu xuống 5178 mẫu, nên split cũ không còn hợp lệ. Split đã được tạo lại thành công và hiện không còn index out of range.

File model `.joblib` trong `experiments/` và `models/` đã được ignore để tránh push binary lớn. Nên push metrics, config, scripts và report; chỉ push model binary nếu team thật sự muốn lưu artifact trên Git.
