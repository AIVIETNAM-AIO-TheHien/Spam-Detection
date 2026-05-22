# Phân tích lỗi LinearSVC

## 1. Tóm tắt kết quả

| Metric | Giá trị |
|---|---:|
| accuracy | 0.9781 |
| macro_f1 | 0.9735 |
| weighted_f1 | 0.9782 |
| spam_precision | 0.9437 |
| spam_recall | 0.9820 |
| spam_f1 | 0.9625 |

## 2. Tổng quan lỗi

| Chỉ số | Giá trị |
|---|---:|
| Tổng số mẫu test | 777 |
| Dự đoán đúng | 760 |
| Tổng số lỗi | 17 |
| Error rate | 2.19% |
| False positive | 13 |
| False negative | 4 |

LinearSVC không có xác suất trực tiếp. Cột `score` là điểm decision_function: score càng cao thì model càng nghiêng về spam.

## 3. Confusion matrix

| Nhãn thật | Dự đoán | Số lượng |
|---|---|---:|
| ham | ham | 542 |
| ham | spam | 13 |
| spam | ham | 4 |
| spam | spam | 218 |

## 4. Đặc điểm các nhóm lỗi

| Nhóm lỗi | Số mẫu | Độ dài TB | Số từ TB | Có URL | Có số | Có keyword spam |
|---|---:|---:|---:|---:|---:|---:|
| false_negative | 4 | 1325.0 | 195.5 | 3 | 4 | 3 |
| false_positive | 13 | 749.7 | 129.7 | 8 | 8 | 8 |

## 5. Ví dụ false positive

| # | score | True | Pred | URL | Số | Keyword | Nội dung rút gọn |
|---:|---:|---|---|---|---|---|---|
| 1 | -0.1660 | ham | spam | có | có | có | cancelling earthlink sure you call their customer service number they can unsubscribe you took 4 tries took 5 make sure you get confirmation number does that help? had 5 of those lovely company max URL |
| 2 | -0.0108 | ham | spam | có | có | có | iiu nameserver problems four of the nameservers are publishing out of date dns information the four listed below are publishing the zone from last friday they are six days behind reality 192 16 202 11 net 192 93 0 4 ns2... |
| 3 | 0.4469 | ham | spam | có | có | có | ashfield online 9 1 2002 aric mckeown ashfield online 2002 aric mckeown comic for sunday september 1 2002 powered flash keenspot comic visit the ashfield online website at URL ! click here URL note you not see comic in t... |
| 4 | 0.5816 | ham | spam | không | không | không | asteroids anyone this multi part message in mime format |
| 5 | -0.1637 | ham | spam | không | có | không | sadev razor users mutating spam justin mason wrote interestingly some of these seem apparently to encrypted versions of the recipient email address to see this rot13 address and grep your spam archive there at least 1 hi... |
| 6 | -0.1575 | ham | spam | có | có | có | cancelling earthlink joel warren wrote part 1 1 type plain text text plain encoding quoted printable hey greg! name joel warren and need to cancel earthlink your message was posted awhile ago not sure you can still refer... |
| 7 | -0.1334 | ham | spam | có | có | có | brain hurts more than dozen jokes thanks guys and girls! plus some anti french abuse from the usual suspect well brain doesn hurt much any more and was well worth now got even faster method for elliptic curve point count... |
| 8 | 0.8315 | ham | spam | không | không | không | hi! new here this multi part message in mime format |

## 6. Ví dụ false negative

| # | score | True | Pred | URL | Số | Keyword | Nội dung rút gọn |
|---:|---:|---|---|---|---|---|---|
| 1 | -0.5147 | spam | ham | có | có | có | the gold buddha in 1957 monastery in thailand was being relocated and group of monks was put in charge of moving giant clay buddha in the midst of the move one of the monks noticed crack in the buddha concerned about dam... |
| 2 | -0.2130 | spam | ham | có | có | có | sps odrekao slobodana milosevica socijalisticka partija srbije predvodjena grupom starih socijalista celu milomirem minicem konacno odrekla politike herojskog drzanja predsednika stranke slobodana milosevica saopstenjima... |
| 3 | -0.2596 | spam | ham | không | có | không | the mighty pro grill ! adv 2147pbqg 8 |
| 4 | -0.4620 | spam | ham | có | có | có | discreet penis enlargement 4623 ! URL ! URL 4623 |

## 7. File chi tiết

- Các mẫu sai: `prediction_errors.csv`
- Confusion matrix dạng CSV: `confusion_matrix.csv`
- Tóm tắt lỗi dạng JSON: `error_summary.json`