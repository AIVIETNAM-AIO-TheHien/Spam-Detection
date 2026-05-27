# Hệ thống phát hiện Spam Email/SMS

Project này xây dựng một hệ thống Machine Learning dùng để phân loại nội dung email hoặc tin nhắn SMS thành hai nhóm: **Spam** và **Ham**. Hệ thống hướng tới việc hỗ trợ người dùng phát hiện các nội dung không mong muốn như tin nhắn rác, quảng cáo, lừa đảo hoặc email có dấu hiệu spam.

Project áp dụng một pipeline xử lý văn bản cơ bản trong NLP, bao gồm tiền xử lý dữ liệu, trích xuất đặc trưng bằng TF-IDF và huấn luyện các mô hình phân loại như Naive Bayes, Logistic Regression và Linear SVM. Ngoài phần huấn luyện và đánh giá mô hình, chúng tôi còn có giao diện demo bằng Streamlit để người dùng nhập nội dung bất kỳ và xem kết quả dự đoán trực tiếp.

Trong quá trình đánh giá, chúng tôi không chỉ quan tâm đến accuracy mà còn phân tích chi tiết các loại lỗi của mô hình. Kết quả cho thấy mô hình có khả năng nhận diện spam khá tốt, nhưng vẫn tồn tại trường hợp email/tin nhắn hợp lệ bị dự đoán nhầm thành spam. Vì vậy, project tập trung trình bày rõ hiệu quả thực tế của từng mô hình thông qua precision, recall, F1-score, confusion matrix và error analysis.

## Tổng quan dự án

Spam detection là bài toán phân loại văn bản nhị phân. Với một email hoặc tin nhắn đầu vào, hệ thống sẽ xử lý văn bản, biến đổi thành đặc trưng TF-IDF, sau đó dùng mô hình Machine Learning để dự đoán nội dung đó là spam hay ham.

Dự án hiện so sánh các mô hình NLP truyền thống:

- Multinomial Naive Bayes
- Logistic Regression
- Linear Support Vector Classifier, viết tắt là LinearSVC

Repo cũng có ứng dụng demo bằng Streamlit để người dùng nhập thử nội dung email hoặc tin nhắn và xem kết quả dự đoán trực tiếp.

## Mục tiêu chính

Project tập trung vào việc xây dựng và đánh giá một pipeline phát hiện spam hoàn chỉnh, từ dữ liệu đầu vào đến kết quả phân loại cuối cùng. Các mục tiêu chính gồm:

1. Xây dựng pipeline xử lý văn bản cho email/SMS, bao gồm preprocessing, TF-IDF vectorization và model prediction.
2. So sánh hiệu quả của các mô hình Machine Learning truyền thống gồm Multinomial Naive Bayes, Logistic Regression và LinearSVC.
3. Đánh giá mô hình bằng nhiều chỉ số thay vì chỉ dựa vào accuracy, bao gồm precision, recall, F1-score, confusion matrix, false positives và false negatives.
4. Phân tích lỗi dự đoán để thấy rõ mô hình thường nhầm ở những trường hợp nào, đặc biệt là các trường hợp ham bị dự đoán thành spam hoặc spam bị bỏ sót.
5. Xây dựng demo đơn giản bằng Streamlit để minh họa luồng hoạt động của hệ thống trên dữ liệu nhập thực tế.

Kết quả hiện tại cho thấy các mô hình đạt hiệu năng tốt trên tập test, đặc biệt là LinearSVC. Tuy nhiên, chúng tôi vẫn ghi nhận một số lỗi dự đoán, trong đó có các trường hợp nội dung hợp lệ bị phân loại nhầm thành spam. Đây là điểm cần tiếp tục cải thiện nếu muốn triển khai hệ thống trong môi trường thực tế.

Khi chọn mô hình theo hướng triển khai thực tế, cần xem xét đồng thời **spam precision**, **spam recall**, **false positive count** và **false negative count**. Việc tăng khả năng bắt spam có thể làm mô hình dự đoán nhầm nhiều ham thành spam hơn, nên cần phân tích trade-off thay vì chỉ chọn mô hình có F1-score cao nhất.

## Pipeline xử lý

```text
Raw email / message
        ↓
Preprocessing
        ↓
TF-IDF vectorization
        ↓
Model prediction
        ↓
Spam / Ham output
```

### Các bước preprocessing

Pipeline tiền xử lý bao gồm:

- Chuyển chữ về lowercase
- Chuẩn hóa URL, email, số điện thoại và tiền tệ
- Loại bỏ nhiễu hệ thống như email headers, PGP signatures, chuỗi mã hóa dài và technical artifacts
- Loại bỏ boilerplate từ mailing list
- Tùy chọn xử lý dấu câu và chữ số
- Giới hạn độ dài văn bản để tránh input quá dài

Logic preprocessing dùng chung được đặt trong:

```text
src/data/preprocess_fixed.py
```

## Cấu trúc repository

```text
Spam-Detection-main/
├── .github/                    # Template cho issue và pull request
├── configs/                    # File cấu hình YAML/JSON cho thí nghiệm
├── data/                       # Dataset gốc và dataset custom
│   ├── raw/                    # Dataset SMS gốc
│   └── custom/                 # Dataset email đã tiền xử lý
├── docs/                       # Kế hoạch, báo cáo, QA/QC và tài liệu phân tích
├── experiments/                # Kết quả thí nghiệm, metrics, split indices và error analysis
├── models/                     # Model artifacts dùng cho Streamlit demo
├── notebooks/                  # Notebook khám phá dữ liệu, nếu cần
├── scripts/                    # Script chạy training, evaluation, experiment và demo
├── src/                        # Source code tái sử dụng
│   ├── data/                   # Tiền xử lý và kiểm tra dữ liệu
│   ├── evaluation/             # Tính toán metrics
│   ├── models/                 # Thành phần liên quan đến model
│   └── utils/                  # Helper dùng chung
├── tests/                      # Unit tests và kiểm tra tính toàn vẹn của split
├── requirements.txt
└── README.md
```

## Dataset

Repo hiện có hai file dữ liệu chính:

| Đường dẫn | Mục đích |
|---|---|
| `data/raw/Assassin.csv` | Dataset SMS spam/ham gốc |
| `data/custom/Assassin_preprocessed.csv` | Dataset email-style đã tiền xử lý, dùng cho các thí nghiệm hiện tại |

Tóm tắt dataset custom hiện tại:

| Lớp | Nhãn | Số mẫu |
|---|---:|---:|
| Ham | `0` | 3,698 |
| Spam | `1` | 1,480 |
| **Tổng** |  | **5,178** |

Fixed split được lưu tại:

```text
experiments/splits/split_indices.json
```

Tỷ lệ chia hiện tại:

| Split | Số mẫu |
|---|---:|
| Train | 3,624 |
| Dev | 777 |
| Test | 777 |

## Mô hình và kết quả

Tất cả mô hình chính đều dùng TF-IDF features và cùng một fixed train/dev/test split.

| Model | Cách chọn | Test accuracy | Spam precision | Spam recall | Spam F1 | False positives | False negatives |
|---|---|---:|---:|---:|---:|---:|---:|
| Multinomial Naive Bayes | Baseline | 0.9395 | 0.9781 | 0.8063 | 0.8840 | 4 | 43 |
| Logistic Regression | Dev-tuned threshold | 0.9768 | 0.9595 | 0.9595 | 0.9595 | 9 | 9 |
| LinearSVC | Default decision boundary | 0.9833 | 0.9772 | 0.9640 | 0.9705 | 5 | 8 |
| LinearSVC | Dev-tuned F1 threshold | 0.9794 | 0.9478 | 0.9820 | 0.9646 | 12 | 4 |

## Cài đặt

Clone repository và cài đặt dependencies:

```bash
git clone <your-repository-url>
cd Spam-Detection-main
python -m venv .venv
```

Kích hoạt virtual environment:

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

Cài đặt thư viện:

```bash
pip install -r requirements.txt
```

## Cách chạy project

### 1. Chạy test

```bash
pytest -q
```

Kết quả mong đợi:

```text
46 passed
```

### 2. Tạo fixed split

Repo đã có sẵn fixed split. Chỉ chạy lệnh này nếu file `experiments/splits/split_indices.json` chưa tồn tại:

```bash
python scripts/create_split.py
```

### 3. Train và evaluate baseline

```bash
python scripts/train_baseline.py
python scripts/evaluate_baseline.py
```

### 4. Chạy thí nghiệm Logistic Regression

```bash
python scripts/run_logistic_experiments.py
```

### 5. Chạy thí nghiệm LinearSVC

```bash
python scripts/run_linear_svc_experiments.py
```

### 6. Chạy Streamlit demo

```bash
streamlit run scripts/streamlit_app.py
```

Trong demo, người dùng nhập raw email hoặc message. Ứng dụng sẽ hiển thị:

- Kết quả dự đoán Spam/Ham
- Vote của các model
- Độ dài text sau khi xử lý
- So sánh kết quả từng model
- Probability và decision score 
- Model artifact đang dùng và thông tin threshold

## Tài liệu quan trọng trong repo

| Tài liệu | Nội dung |
|---|---|
| `docs/Plan.md` | Kế hoạch nhóm, phân công vai trò và mục tiêu theo tuần |
| `docs/QA_QC.md` | Checklist QA và tiêu chí nghiệm thu |
| `docs/structure.md` | Giải thích cấu trúc repository |
| `docs/folder_workflow.md` | Quy ước sử dụng folder và workflow |
| `docs/feedback_preprocess.md` | Yêu cầu và lý do thiết kế preprocessing |
| `docs/baseline_vs_best_logistic_regression_report.md` | So sánh baseline và Logistic Regression |
| `docs/logistic_regression_vs_linear_svc_report.md` | So sánh Logistic Regression và LinearSVC |
| `docs/datacollection_guide.md` | Gợi ý nguồn dữ liệu và hướng augmentation |

## Version control và cộng tác nhóm

Repo đã có các template hỗ trợ cộng tác trên GitHub:

- Bug report template
- Feature request template
- Pull request template

Workflow khuyến nghị:

1. Tạo branch riêng cho từng feature hoặc bug fix.
2. Chỉ sửa ở các folder liên quan.
3. Mở pull request với mô tả rõ ràng.
4. Đính kèm kết quả thí nghiệm, ảnh demo hoặc test log nếu cần.
5. Chỉ merge sau khi đã được review.

## Hạn chế hiện tại

- Model hiện chủ yếu được huấn luyện trên dữ liệu tiếng Anh dạng SMS/email, nên spam tiếng Việt và phishing pattern mới có thể vẫn là điểm yếu.
- TF-IDF models không hiểu ngữ nghĩa sâu như transformer-based models.
- Việc chọn threshold ảnh hưởng mạnh đến loại lỗi của hệ thống. Threshold thiên về recall có thể bắt được nhiều spam hơn nhưng dễ tăng false positive.
- Một số báo cáo cũ có thể cần được regenerate nếu dataset hoặc split đã thay đổi.

## Hướng cải thiện tiếp theo

- Thêm cách tìm threshold theo hướng precision-oriented, ví dụ tối đa hóa spam precision nhưng vẫn giữ recall trên một mức tối thiểu.
- Bổ sung thêm real-world ham examples để giảm false positive.
- Bổ sung thêm phishing, scam và promotional spam mới.
- Tạo một QA test set nhỏ gồm các edge cases cho cả spam và ham.
- Cập nhật các report để thống nhất với dataset, split và mục tiêu hiện tại của project.
- Cân nhắc cross-validation hoặc holdout test set mới trước khi chọn model cuối cùng.

