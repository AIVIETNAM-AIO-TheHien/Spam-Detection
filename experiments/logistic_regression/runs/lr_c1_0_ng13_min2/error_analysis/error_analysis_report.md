# Phân tích lỗi Logistic Regression

## 1. Tóm tắt kết quả

| Metric | Giá trị |
|---|---:|
| accuracy | 0.9730 |
| macro_f1 | 0.9671 |
| weighted_f1 | 0.9731 |
| spam_precision | 0.9427 |
| spam_recall | 0.9640 |
| spam_f1 | 0.9532 |

## 2. Tổng quan lỗi

| Chỉ số | Giá trị |
|---|---:|
| Tổng số mẫu test | 777 |
| Dự đoán đúng | 756 |
| Tổng số lỗi | 21 |
| Error rate | 2.70% |
| False positive | 13 |
| False negative | 8 |

False positive là mẫu ham bị dự đoán nhầm thành spam. False negative là mẫu spam bị bỏ sót và dự đoán thành ham.

## 3. Confusion matrix

| Nhãn thật | Dự đoán | Số lượng |
|---|---|---:|
| ham | ham | 542 |
| ham | spam | 13 |
| spam | ham | 8 |
| spam | spam | 214 |

## 4. Đặc điểm các nhóm lỗi

| Nhóm lỗi | Số mẫu | Độ dài TB | Số từ TB | Có URL | Có số | Có keyword spam |
|---|---:|---:|---:|---:|---:|---:|
| false_negative | 8 | 1197.2 | 188.8 | 7 | 7 | 7 |
| false_positive | 13 | 1241.4 | 209.0 | 6 | 9 | 8 |

## 5. Ví dụ false positive

| # | P(spam) | True | Pred | URL | Số | Keyword | Nội dung rút gọn |
|---:|---:|---|---|---|---|---|---|
| 1 | 0.5834 | ham | spam | không | có | có | take all your music on the road zdnet shopper buyer alert july 12 2002 when ordering make sure the reseller manufacturer provides the rebate coupon you need expires july 5 2002 $50 off cidco mailstation with subscription... |
| 2 | 0.6165 | ham | spam | có | có | có | iiu nameserver problems four of the nameservers are publishing out of date dns information the four listed below are publishing the zone from last friday they are six days behind reality 192 16 202 11 net 192 93 0 4 ns2... |
| 3 | 0.8626 | ham | spam | có | có | có | ashfield online 9 1 2002 aric mckeown ashfield online 2002 aric mckeown comic for sunday september 1 2002 powered flash keenspot comic visit the ashfield online website at URL ! click here URL note you not see comic in t... |
| 4 | 0.8686 | ham | spam | không | không | không | asteroids anyone this multi part message in mime format |
| 5 | 0.4948 | ham | spam | không | có | không | sadev razor users mutating spam justin mason wrote interestingly some of these seem apparently to encrypted versions of the recipient email address to see this rot13 address and grep your spam archive there at least 1 hi... |
| 6 | 0.4769 | ham | spam | có | có | có | cancelling earthlink joel warren wrote part 1 1 type plain text text plain encoding quoted printable hey greg! name joel warren and need to cancel earthlink your message was posted awhile ago not sure you can still refer... |
| 7 | 0.4865 | ham | spam | có | có | có | brain hurts more than dozen jokes thanks guys and girls! plus some anti french abuse from the usual suspect well brain doesn hurt much any more and was well worth now got even faster method for elliptic curve point count... |
| 8 | 0.9162 | ham | spam | không | không | không | hi! new here this multi part message in mime format |

Nhận xét nhanh: false positive thường là các email ham có hình thức giống spam, ví dụ newsletter, thông báo có nhiều link, giá tiền, khuyến mãi, hoặc MIME/html.

## 6. Ví dụ false negative

| # | P(spam) | True | Pred | URL | Số | Keyword | Nội dung rút gọn |
|---:|---:|---|---|---|---|---|---|
| 1 | 0.2230 | spam | ham | có | có | có | the gold buddha in 1957 monastery in thailand was being relocated and group of monks was put in charge of moving giant clay buddha in the midst of the move one of the monks noticed crack in the buddha concerned about dam... |
| 2 | 0.4167 | spam | ham | có | không | có | scoop liseli kayit this net email sponsored thinkgeek welcome to geek heaven URL sitescooper talk mailing list EMAIL URL |
| 3 | 0.3606 | spam | ham | có | có | có | sps odrekao slobodana milosevica socijalisticka partija srbije predvodjena grupom starih socijalista celu milomirem minicem konacno odrekla politike herojskog drzanja predsednika stranke slobodana milosevica saopstenjima... |
| 4 | 0.3513 | spam | ham | không | có | không | the mighty pro grill ! adv 2147pbqg 8 |
| 5 | 0.3327 | spam | ham | có | có | có | ceviri yazilimlari html nternette kendnz daha hssedeceksnz nglzce web sayfalarinin trke evr yazilimi $ 39 kdv nglzce den trke blgsayar destekl metn evr yazilimi word altinda kullanim windows versyonlari uyumlu $ 69 kdv y... |
| 6 | 0.2390 | spam | ham | có | có | có | discreet penis enlargement 4623 ! URL ! URL 4623 |
| 7 | 0.3004 | spam | ham | có | có | có | scoop ceviri yazilimlari html nternette kendnz daha hssedeceksnz nglzce web sayfalarinin trke evr yazilimi $ 39 kdv nglzce den trke blgsayar destekl metn evr yazilimi word altinda kullanim windows versyonlari uyumlu $ 69... |
| 8 | 0.3771 | spam | ham | có | có | có | dcms dev inheritance date may 16 2002 EMAIL dear sir eduado mello one of the principal commanders of the union for the total independence of angola unita well needless telling very long story here for our story indeed in... |

Nhận xét nhanh: false negative là nhóm cần ưu tiên đọc kỹ, vì đây là spam bị bỏ sót. Nếu các mẫu này thiếu keyword spam rõ ràng, cần xem xét thêm feature về URL, phone, money token, ngôn ngữ lạ, hoặc bổ sung data cùng pattern.

## 7. File chi tiết

- Các mẫu sai: `prediction_errors.csv`
- Confusion matrix dạng CSV: `confusion_matrix.csv`
- Tóm tắt lỗi dạng JSON: `error_summary.json`