# Phân tích lỗi Logistic Regression

## 1. Tóm tắt kết quả

| Metric | Giá trị |
|---|---:|
| accuracy | 0.9678 |
| macro_f1 | 0.9608 |
| weighted_f1 | 0.9679 |
| spam_precision | 0.9339 |
| spam_recall | 0.9550 |
| spam_f1 | 0.9443 |

## 2. Tổng quan lỗi

| Chỉ số | Giá trị |
|---|---:|
| Tổng số mẫu test | 777 |
| Dự đoán đúng | 752 |
| Tổng số lỗi | 25 |
| Error rate | 3.22% |
| False positive | 15 |
| False negative | 10 |

False positive là mẫu ham bị dự đoán nhầm thành spam. False negative là mẫu spam bị bỏ sót và dự đoán thành ham.

## 3. Confusion matrix

| Nhãn thật | Dự đoán | Số lượng |
|---|---|---:|
| ham | ham | 540 |
| ham | spam | 15 |
| spam | ham | 10 |
| spam | spam | 212 |

## 4. Đặc điểm các nhóm lỗi

| Nhóm lỗi | Số mẫu | Độ dài TB | Số từ TB | Có URL | Có số | Có keyword spam |
|---|---:|---:|---:|---:|---:|---:|
| false_negative | 10 | 987.6 | 157.1 | 8 | 8 | 8 |
| false_positive | 15 | 1814.7 | 309.5 | 8 | 11 | 10 |

## 5. Ví dụ false positive

| # | P(spam) | True | Pred | URL | Số | Keyword | Nội dung rút gọn |
|---:|---:|---|---|---|---|---|---|
| 1 | 0.6361 | ham | spam | không | có | có | take all your music on the road zdnet shopper buyer alert july 12 2002 when ordering make sure the reseller manufacturer provides the rebate coupon you need expires july 5 2002 $50 off cidco mailstation with subscription... |
| 2 | 0.4996 | ham | spam | có | có | có | sacvs cvs spamassassin manifest 1 100 2 11 1 100 2 12 update of in directory usw cvs1 tmp cvs serv6501 modified files tag 4 0 manifest log message removed old evolver index manifest rcs file retrieving revision 1 100 2 1... |
| 3 | 0.6187 | ham | spam | có | có | có | iiu nameserver problems four of the nameservers are publishing out of date dns information the four listed below are publishing the zone from last friday they are six days behind reality 192 16 202 11 net 192 93 0 4 ns2... |
| 4 | 0.8871 | ham | spam | có | có | có | ashfield online 9 1 2002 aric mckeown ashfield online 2002 aric mckeown comic for sunday september 1 2002 powered flash keenspot comic visit the ashfield online website at URL ! click here URL note you not see comic in t... |
| 5 | 0.8515 | ham | spam | không | không | không | asteroids anyone this multi part message in mime format |
| 6 | 0.4897 | ham | spam | không | có | không | sadev razor users mutating spam justin mason wrote interestingly some of these seem apparently to encrypted versions of the recipient email address to see this rot13 address and grep your spam archive there at least 1 hi... |
| 7 | 0.4745 | ham | spam | có | có | có | ilug dermot beirne dublin exel out of the office will out of the office starting 02 08 2002 and will not return until 06 08 2002 out of the office until tuesday 6th august will reply to messages on return thank you dermo... |
| 8 | 0.4680 | ham | spam | có | có | có | cancelling earthlink joel warren wrote part 1 1 type plain text text plain encoding quoted printable hey greg! name joel warren and need to cancel earthlink your message was posted awhile ago not sure you can still refer... |

Nhận xét nhanh: false positive thường là các email ham có hình thức giống spam, ví dụ newsletter, thông báo có nhiều link, giá tiền, khuyến mãi, hoặc MIME/html.

## 6. Ví dụ false negative

| # | P(spam) | True | Pred | URL | Số | Keyword | Nội dung rút gọn |
|---:|---:|---|---|---|---|---|---|
| 1 | 0.4595 | spam | ham | không | không | không | get on the bus gbvzzzz click here to see the bang bus wild!!!!!! tepyycemkckiflbsvpcyi |
| 2 | 0.2260 | spam | ham | có | có | có | the gold buddha in 1957 monastery in thailand was being relocated and group of monks was put in charge of moving giant clay buddha in the midst of the move one of the monks noticed crack in the buddha concerned about dam... |
| 3 | 0.3397 | spam | ham | có | có | có | hassle free microsoft sql server remote database administration ! URL ! URL ! URL ! URL ! URL ! URL ! URL ! URL ! URL ! URL ! URL visit at URL call at 214 740 0923 to removed reply with remove in the subject line |
| 4 | 0.4314 | spam | ham | có | không | có | scoop liseli kayit this net email sponsored thinkgeek welcome to geek heaven URL sitescooper talk mailing list EMAIL URL |
| 5 | 0.3402 | spam | ham | có | có | có | sps odrekao slobodana milosevica socijalisticka partija srbije predvodjena grupom starih socijalista celu milomirem minicem konacno odrekla politike herojskog drzanja predsednika stranke slobodana milosevica saopstenjima... |
| 6 | 0.3242 | spam | ham | không | có | không | the mighty pro grill ! adv 2147pbqg 8 |
| 7 | 0.4393 | spam | ham | có | có | có | ceviri yazilimlari html nternette kendnz daha hssedeceksnz nglzce web sayfalarinin trke evr yazilimi $ 39 kdv nglzce den trke blgsayar destekl metn evr yazilimi word altinda kullanim windows versyonlari uyumlu $ 69 kdv y... |
| 8 | 0.2573 | spam | ham | có | có | có | discreet penis enlargement 4623 ! URL ! URL 4623 |

Nhận xét nhanh: false negative là nhóm cần ưu tiên đọc kỹ, vì đây là spam bị bỏ sót. Nếu các mẫu này thiếu keyword spam rõ ràng, cần xem xét thêm feature về URL, phone, money token, ngôn ngữ lạ, hoặc bổ sung data cùng pattern.

## 7. File chi tiết

- Các mẫu sai: `prediction_errors.csv`
- Confusion matrix dạng CSV: `confusion_matrix.csv`
- Tóm tắt lỗi dạng JSON: `error_summary.json`