# Phân tích lỗi LinearSVC

## 1. Tóm tắt kết quả

| Metric | Giá trị |
|---|---:|
| accuracy | 0.9781 |
| macro_f1 | 0.9734 |
| weighted_f1 | 0.9782 |
| spam_precision | 0.9476 |
| spam_recall | 0.9775 |
| spam_f1 | 0.9623 |

## 2. Tổng quan lỗi

| Chỉ số | Giá trị |
|---|---:|
| Tổng số mẫu test | 777 |
| Dự đoán đúng | 760 |
| Tổng số lỗi | 17 |
| Error rate | 2.19% |
| False positive | 12 |
| False negative | 5 |

LinearSVC không có xác suất trực tiếp. Cột `score` là điểm decision_function: score càng cao thì model càng nghiêng về spam.

## 3. Confusion matrix

| Nhãn thật | Dự đoán | Số lượng |
|---|---|---:|
| ham | ham | 543 |
| ham | spam | 12 |
| spam | ham | 5 |
| spam | spam | 217 |

## 4. Đặc điểm các nhóm lỗi

| Nhóm lỗi | Số mẫu | Độ dài TB | Số từ TB | Có URL | Có số | Có keyword spam |
|---|---:|---:|---:|---:|---:|---:|
| false_negative | 5 | 1077.2 | 159.2 | 3 | 4 | 3 |
| false_positive | 12 | 718.4 | 122.1 | 8 | 7 | 8 |

## 5. Ví dụ false positive

| # | score | True | Pred | URL | Số | Keyword | Nội dung rút gọn |
|---:|---:|---|---|---|---|---|---|
| 1 | -0.1393 | ham | spam | có | có | có | cancelling earthlink sure you call their customer service number they can unsubscribe you took 4 tries took 5 make sure you get confirmation number does that help? had 5 of those lovely company max URL |
| 2 | -0.0846 | ham | spam | có | có | có | iiu nameserver problems four of the nameservers are publishing out of date dns information the four listed below are publishing the zone from last friday they are six days behind reality 192 16 202 11 net 192 93 0 4 ns2... |
| 3 | 0.3602 | ham | spam | có | có | có | ashfield online 9 1 2002 aric mckeown ashfield online 2002 aric mckeown comic for sunday september 1 2002 powered flash keenspot comic visit the ashfield online website at URL ! click here URL note you not see comic in t... |
| 4 | 0.4691 | ham | spam | không | không | không | asteroids anyone this multi part message in mime format |
| 5 | -0.2121 | ham | spam | có | có | có | cancelling earthlink joel warren wrote part 1 1 type plain text text plain encoding quoted printable hey greg! name joel warren and need to cancel earthlink your message was posted awhile ago not sure you can still refer... |
| 6 | -0.1329 | ham | spam | có | có | có | brain hurts more than dozen jokes thanks guys and girls! plus some anti french abuse from the usual suspect well brain doesn hurt much any more and was well worth now got even faster method for elliptic curve point count... |
| 7 | 0.7556 | ham | spam | không | không | không | hi! new here this multi part message in mime format |
| 8 | -0.1454 | ham | spam | có | không | có | ilug dial on demand could you please help how to set dial on demand what are the packages needed and other requirements to get on you know site that has how to and the package where to get please help please for this was... |

## 6. Ví dụ false negative

| # | score | True | Pred | URL | Số | Keyword | Nội dung rút gọn |
|---:|---:|---|---|---|---|---|---|
| 1 | -0.2654 | spam | ham | không | không | không | get on the bus gbvzzzz click here to see the bang bus wild!!!!!! tepyycemkckiflbsvpcyi |
| 2 | -0.5294 | spam | ham | có | có | có | the gold buddha in 1957 monastery in thailand was being relocated and group of monks was put in charge of moving giant clay buddha in the midst of the move one of the monks noticed crack in the buddha concerned about dam... |
| 3 | -0.2717 | spam | ham | có | có | có | sps odrekao slobodana milosevica socijalisticka partija srbije predvodjena grupom starih socijalista celu milomirem minicem konacno odrekla politike herojskog drzanja predsednika stranke slobodana milosevica saopstenjima... |
| 4 | -0.3270 | spam | ham | không | có | không | the mighty pro grill ! adv 2147pbqg 8 |
| 5 | -0.4069 | spam | ham | có | có | có | discreet penis enlargement 4623 ! URL ! URL 4623 |

## 7. File chi tiết

- Các mẫu sai: `prediction_errors.csv`
- Confusion matrix dạng CSV: `confusion_matrix.csv`
- Tóm tắt lỗi dạng JSON: `error_summary.json`