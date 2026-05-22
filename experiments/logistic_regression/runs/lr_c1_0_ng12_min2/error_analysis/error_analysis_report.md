# Phân tích lỗi Logistic Regression

## 1. Tóm tắt kết quả

| Metric | Giá trị |
|---|---:|
| accuracy | 0.9768 |
| macro_f1 | 0.9716 |
| weighted_f1 | 0.9768 |
| spam_precision | 0.9595 |
| spam_recall | 0.9595 |
| spam_f1 | 0.9595 |

## 2. Tổng quan lỗi

| Chỉ số | Giá trị |
|---|---:|
| Tổng số mẫu test | 777 |
| Dự đoán đúng | 759 |
| Tổng số lỗi | 18 |
| Error rate | 2.32% |
| False positive | 9 |
| False negative | 9 |

False positive là mẫu ham bị dự đoán nhầm thành spam. False negative là mẫu spam bị bỏ sót và dự đoán thành ham.

## 3. Confusion matrix

| Nhãn thật | Dự đoán | Số lượng |
|---|---|---:|
| ham | ham | 546 |
| ham | spam | 9 |
| spam | ham | 9 |
| spam | spam | 213 |

## 4. Đặc điểm các nhóm lỗi

| Nhóm lỗi | Số mẫu | Độ dài TB | Số từ TB | Có URL | Có số | Có keyword spam |
|---|---:|---:|---:|---:|---:|---:|
| false_negative | 9 | 1084.0 | 172.4 | 7 | 8 | 7 |
| false_positive | 9 | 1330.9 | 222.4 | 3 | 5 | 5 |

## 5. Ví dụ false positive

| # | P(spam) | True | Pred | URL | Số | Keyword | Nội dung rút gọn |
|---:|---:|---|---|---|---|---|---|
| 1 | 0.6294 | ham | spam | không | có | có | take all your music on the road zdnet shopper buyer alert july 12 2002 when ordering make sure the reseller manufacturer provides the rebate coupon you need expires july 5 2002 $50 off cidco mailstation with subscription... |
| 2 | 0.6155 | ham | spam | có | có | có | iiu nameserver problems four of the nameservers are publishing out of date dns information the four listed below are publishing the zone from last friday they are six days behind reality 192 16 202 11 net 192 93 0 4 ns2... |
| 3 | 0.8757 | ham | spam | có | có | có | ashfield online 9 1 2002 aric mckeown ashfield online 2002 aric mckeown comic for sunday september 1 2002 powered flash keenspot comic visit the ashfield online website at URL ! click here URL note you not see comic in t... |
| 4 | 0.8430 | ham | spam | không | không | không | asteroids anyone this multi part message in mime format |
| 5 | 0.9059 | ham | spam | không | không | không | hi! new here this multi part message in mime format |
| 6 | 0.5162 | ham | spam | có | có | có | vulnwatch foundstone labs advisory information leakage in orinoco and compaq access points foundstone labs advisory apil advisory name information leakage in orinoco and compaq access points release date august 9th 2002... |
| 7 | 0.8656 | ham | spam | không | không | không | inspiring article howard jonas who founded idt this multi part message in mime format |
| 8 | 0.6293 | ham | spam | không | có | có | personal laser picks for students soho users and more! zdnet shopper buyer alert july 16 2002 when ordering make sure the reseller manufacturer provides the rebate coupon you need expires july 5 2002 $50 off cidco mailst... |

Nhận xét nhanh: false positive thường là các email ham có hình thức giống spam, ví dụ newsletter, thông báo có nhiều link, giá tiền, khuyến mãi, hoặc MIME/html.

## 6. Ví dụ false negative

| # | P(spam) | True | Pred | URL | Số | Keyword | Nội dung rút gọn |
|---:|---:|---|---|---|---|---|---|
| 1 | 0.4491 | spam | ham | không | không | không | get on the bus gbvzzzz click here to see the bang bus wild!!!!!! tepyycemkckiflbsvpcyi |
| 2 | 0.2179 | spam | ham | có | có | có | the gold buddha in 1957 monastery in thailand was being relocated and group of monks was put in charge of moving giant clay buddha in the midst of the move one of the monks noticed crack in the buddha concerned about dam... |
| 3 | 0.3273 | spam | ham | có | có | có | hassle free microsoft sql server remote database administration ! URL ! URL ! URL ! URL ! URL ! URL ! URL ! URL ! URL ! URL ! URL visit at URL call at 214 740 0923 to removed reply with remove in the subject line |
| 4 | 0.3432 | spam | ham | có | có | có | sps odrekao slobodana milosevica socijalisticka partija srbije predvodjena grupom starih socijalista celu milomirem minicem konacno odrekla politike herojskog drzanja predsednika stranke slobodana milosevica saopstenjima... |
| 5 | 0.3190 | spam | ham | không | có | không | the mighty pro grill ! adv 2147pbqg 8 |
| 6 | 0.4260 | spam | ham | có | có | có | ceviri yazilimlari html nternette kendnz daha hssedeceksnz nglzce web sayfalarinin trke evr yazilimi $ 39 kdv nglzce den trke blgsayar destekl metn evr yazilimi word altinda kullanim windows versyonlari uyumlu $ 69 kdv y... |
| 7 | 0.2514 | spam | ham | có | có | có | discreet penis enlargement 4623 ! URL ! URL 4623 |
| 8 | 0.4106 | spam | ham | có | có | có | scoop ceviri yazilimlari html nternette kendnz daha hssedeceksnz nglzce web sayfalarinin trke evr yazilimi $ 39 kdv nglzce den trke blgsayar destekl metn evr yazilimi word altinda kullanim windows versyonlari uyumlu $ 69... |

Nhận xét nhanh: false negative là nhóm cần ưu tiên đọc kỹ, vì đây là spam bị bỏ sót. Nếu các mẫu này thiếu keyword spam rõ ràng, cần xem xét thêm feature về URL, phone, money token, ngôn ngữ lạ, hoặc bổ sung data cùng pattern.

## 7. File chi tiết

- Các mẫu sai: `prediction_errors.csv`
- Confusion matrix dạng CSV: `confusion_matrix.csv`
- Tóm tắt lỗi dạng JSON: `error_summary.json`