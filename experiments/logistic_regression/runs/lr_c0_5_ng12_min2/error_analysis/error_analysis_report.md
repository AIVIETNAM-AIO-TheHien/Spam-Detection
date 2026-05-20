# Phân tích lỗi Logistic Regression

## 1. Tóm tắt kết quả

| Metric | Giá trị |
|---|---:|
| accuracy | 0.9717 |
| macro_f1 | 0.9652 |
| weighted_f1 | 0.9716 |
| spam_precision | 0.9545 |
| spam_recall | 0.9459 |
| spam_f1 | 0.9502 |

## 2. Tổng quan lỗi

| Chỉ số | Giá trị |
|---|---:|
| Tổng số mẫu test | 777 |
| Dự đoán đúng | 755 |
| Tổng số lỗi | 22 |
| Error rate | 2.83% |
| False positive | 10 |
| False negative | 12 |

False positive là mẫu ham bị dự đoán nhầm thành spam. False negative là mẫu spam bị bỏ sót và dự đoán thành ham.

## 3. Confusion matrix

| Nhãn thật | Dự đoán | Số lượng |
|---|---|---:|
| ham | ham | 545 |
| ham | spam | 10 |
| spam | ham | 12 |
| spam | spam | 210 |

## 4. Đặc điểm các nhóm lỗi

| Nhóm lỗi | Số mẫu | Độ dài TB | Số từ TB | Có URL | Có số | Có keyword spam |
|---|---:|---:|---:|---:|---:|---:|
| false_negative | 12 | 1098.5 | 175.2 | 10 | 10 | 10 |
| false_positive | 10 | 1261.6 | 212.0 | 4 | 6 | 6 |

## 5. Ví dụ false positive

| # | P(spam) | True | Pred | URL | Số | Keyword | Nội dung rút gọn |
|---:|---:|---|---|---|---|---|---|
| 1 | 0.6564 | ham | spam | không | có | có | take all your music on the road zdnet shopper buyer alert july 12 2002 when ordering make sure the reseller manufacturer provides the rebate coupon you need expires july 5 2002 $50 off cidco mailstation with subscription... |
| 2 | 0.5198 | ham | spam | có | có | có | sacvs cvs spamassassin manifest 1 100 2 11 1 100 2 12 update of in directory usw cvs1 tmp cvs serv6501 modified files tag 4 0 manifest log message removed old evolver index manifest rcs file retrieving revision 1 100 2 1... |
| 3 | 0.6244 | ham | spam | có | có | có | iiu nameserver problems four of the nameservers are publishing out of date dns information the four listed below are publishing the zone from last friday they are six days behind reality 192 16 202 11 net 192 93 0 4 ns2... |
| 4 | 0.8368 | ham | spam | có | có | có | ashfield online 9 1 2002 aric mckeown ashfield online 2002 aric mckeown comic for sunday september 1 2002 powered flash keenspot comic visit the ashfield online website at URL ! click here URL note you not see comic in t... |
| 5 | 0.8221 | ham | spam | không | không | không | asteroids anyone this multi part message in mime format |
| 6 | 0.8815 | ham | spam | không | không | không | hi! new here this multi part message in mime format |
| 7 | 0.5403 | ham | spam | có | có | có | vulnwatch foundstone labs advisory information leakage in orinoco and compaq access points foundstone labs advisory apil advisory name information leakage in orinoco and compaq access points release date august 9th 2002... |
| 8 | 0.8415 | ham | spam | không | không | không | inspiring article howard jonas who founded idt this multi part message in mime format |

Nhận xét nhanh: false positive thường là các email ham có hình thức giống spam, ví dụ newsletter, thông báo có nhiều link, giá tiền, khuyến mãi, hoặc MIME/html.

## 6. Ví dụ false negative

| # | P(spam) | True | Pred | URL | Số | Keyword | Nội dung rút gọn |
|---:|---:|---|---|---|---|---|---|
| 1 | 0.4599 | spam | ham | không | không | không | get on the bus gbvzzzz click here to see the bang bus wild!!!!!! tepyycemkckiflbsvpcyi |
| 2 | 0.2625 | spam | ham | có | có | có | the gold buddha in 1957 monastery in thailand was being relocated and group of monks was put in charge of moving giant clay buddha in the midst of the move one of the monks noticed crack in the buddha concerned about dam... |
| 3 | 0.3393 | spam | ham | có | có | có | hassle free microsoft sql server remote database administration ! URL ! URL ! URL ! URL ! URL ! URL ! URL ! URL ! URL ! URL ! URL visit at URL call at 214 740 0923 to removed reply with remove in the subject line |
| 4 | 0.4990 | spam | ham | có | có | có | the stock has wow factor china rapid economic growth ranked the world bank ten times faster than the world economy the only thing investors should asking themselves how can participate in this explosion? have the answer... |
| 5 | 0.4363 | spam | ham | có | không | có | scoop liseli kayit this net email sponsored thinkgeek welcome to geek heaven URL sitescooper talk mailing list EMAIL URL |
| 6 | 0.3660 | spam | ham | có | có | có | sps odrekao slobodana milosevica socijalisticka partija srbije predvodjena grupom starih socijalista celu milomirem minicem konacno odrekla politike herojskog drzanja predsednika stranke slobodana milosevica saopstenjima... |
| 7 | 0.3500 | spam | ham | không | có | không | the mighty pro grill ! adv 2147pbqg 8 |
| 8 | 0.4767 | spam | ham | có | có | có | ilug social webcam on babe! hello name kelly 18 year old swimming instructor from longbeach intelligent sexy and discreet love to laugh and have fun like most young girls like to out with friends and dance all night when... |

Nhận xét nhanh: false negative là nhóm cần ưu tiên đọc kỹ, vì đây là spam bị bỏ sót. Nếu các mẫu này thiếu keyword spam rõ ràng, cần xem xét thêm feature về URL, phone, money token, ngôn ngữ lạ, hoặc bổ sung data cùng pattern.

## 7. File chi tiết

- Các mẫu sai: `prediction_errors.csv`
- Confusion matrix dạng CSV: `confusion_matrix.csv`
- Tóm tắt lỗi dạng JSON: `error_summary.json`