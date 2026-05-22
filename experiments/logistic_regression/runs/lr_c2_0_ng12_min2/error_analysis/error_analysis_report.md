# Phân tích lỗi Logistic Regression

## 1. Tóm tắt kết quả

| Metric | Giá trị |
|---|---:|
| accuracy | 0.9743 |
| macro_f1 | 0.9686 |
| weighted_f1 | 0.9743 |
| spam_precision | 0.9509 |
| spam_recall | 0.9595 |
| spam_f1 | 0.9552 |

## 2. Tổng quan lỗi

| Chỉ số | Giá trị |
|---|---:|
| Tổng số mẫu test | 777 |
| Dự đoán đúng | 757 |
| Tổng số lỗi | 20 |
| Error rate | 2.57% |
| False positive | 11 |
| False negative | 9 |

False positive là mẫu ham bị dự đoán nhầm thành spam. False negative là mẫu spam bị bỏ sót và dự đoán thành ham.

## 3. Confusion matrix

| Nhãn thật | Dự đoán | Số lượng |
|---|---|---:|
| ham | ham | 544 |
| ham | spam | 11 |
| spam | ham | 9 |
| spam | spam | 213 |

## 4. Đặc điểm các nhóm lỗi

| Nhóm lỗi | Số mẫu | Độ dài TB | Số từ TB | Có URL | Có số | Có keyword spam |
|---|---:|---:|---:|---:|---:|---:|
| false_negative | 9 | 1084.0 | 172.4 | 7 | 8 | 7 |
| false_positive | 11 | 1302.0 | 216.1 | 5 | 7 | 7 |

## 5. Ví dụ false positive

| # | P(spam) | True | Pred | URL | Số | Keyword | Nội dung rút gọn |
|---:|---:|---|---|---|---|---|---|
| 1 | 0.5600 | ham | spam | không | có | có | take all your music on the road zdnet shopper buyer alert july 12 2002 when ordering make sure the reseller manufacturer provides the rebate coupon you need expires july 5 2002 $50 off cidco mailstation with subscription... |
| 2 | 0.5942 | ham | spam | có | có | có | iiu nameserver problems four of the nameservers are publishing out of date dns information the four listed below are publishing the zone from last friday they are six days behind reality 192 16 202 11 net 192 93 0 4 ns2... |
| 3 | 0.8985 | ham | spam | có | có | có | ashfield online 9 1 2002 aric mckeown ashfield online 2002 aric mckeown comic for sunday september 1 2002 powered flash keenspot comic visit the ashfield online website at URL ! click here URL note you not see comic in t... |
| 4 | 0.8537 | ham | spam | không | không | không | asteroids anyone this multi part message in mime format |
| 5 | 0.4520 | ham | spam | có | có | có | brain hurts more than dozen jokes thanks guys and girls! plus some anti french abuse from the usual suspect well brain doesn hurt much any more and was well worth now got even faster method for elliptic curve point count... |
| 6 | 0.9211 | ham | spam | không | không | không | hi! new here this multi part message in mime format |
| 7 | 0.4846 | ham | spam | có | có | có | vulnwatch foundstone labs advisory information leakage in orinoco and compaq access points foundstone labs advisory apil advisory name information leakage in orinoco and compaq access points release date august 9th 2002... |
| 8 | 0.8810 | ham | spam | không | không | không | inspiring article howard jonas who founded idt this multi part message in mime format |

Nhận xét nhanh: false positive thường là các email ham có hình thức giống spam, ví dụ newsletter, thông báo có nhiều link, giá tiền, khuyến mãi, hoặc MIME/html.

## 6. Ví dụ false negative

| # | P(spam) | True | Pred | URL | Số | Keyword | Nội dung rút gọn |
|---:|---:|---|---|---|---|---|---|
| 1 | 0.4326 | spam | ham | không | không | không | get on the bus gbvzzzz click here to see the bang bus wild!!!!!! tepyycemkckiflbsvpcyi |
| 2 | 0.1772 | spam | ham | có | có | có | the gold buddha in 1957 monastery in thailand was being relocated and group of monks was put in charge of moving giant clay buddha in the midst of the move one of the monks noticed crack in the buddha concerned about dam... |
| 3 | 0.3225 | spam | ham | có | có | có | hassle free microsoft sql server remote database administration ! URL ! URL ! URL ! URL ! URL ! URL ! URL ! URL ! URL ! URL ! URL visit at URL call at 214 740 0923 to removed reply with remove in the subject line |
| 4 | 0.3198 | spam | ham | có | có | có | sps odrekao slobodana milosevica socijalisticka partija srbije predvodjena grupom starih socijalista celu milomirem minicem konacno odrekla politike herojskog drzanja predsednika stranke slobodana milosevica saopstenjima... |
| 5 | 0.2912 | spam | ham | không | có | không | the mighty pro grill ! adv 2147pbqg 8 |
| 6 | 0.4402 | spam | ham | có | có | có | ceviri yazilimlari html nternette kendnz daha hssedeceksnz nglzce web sayfalarinin trke evr yazilimi $ 39 kdv nglzce den trke blgsayar destekl metn evr yazilimi word altinda kullanim windows versyonlari uyumlu $ 69 kdv y... |
| 7 | 0.2237 | spam | ham | có | có | có | discreet penis enlargement 4623 ! URL ! URL 4623 |
| 8 | 0.4120 | spam | ham | có | có | có | scoop ceviri yazilimlari html nternette kendnz daha hssedeceksnz nglzce web sayfalarinin trke evr yazilimi $ 39 kdv nglzce den trke blgsayar destekl metn evr yazilimi word altinda kullanim windows versyonlari uyumlu $ 69... |

Nhận xét nhanh: false negative là nhóm cần ưu tiên đọc kỹ, vì đây là spam bị bỏ sót. Nếu các mẫu này thiếu keyword spam rõ ràng, cần xem xét thêm feature về URL, phone, money token, ngôn ngữ lạ, hoặc bổ sung data cùng pattern.

## 7. File chi tiết

- Các mẫu sai: `prediction_errors.csv`
- Confusion matrix dạng CSV: `confusion_matrix.csv`
- Tóm tắt lỗi dạng JSON: `error_summary.json`