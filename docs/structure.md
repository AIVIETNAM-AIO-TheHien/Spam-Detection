# Cấu trúc dự án

Tài liệu này mô tả cấu trúc hiện tại của repo sau khi đã dọn lại theo chức năng. Mục tiêu là tách rõ code, dữ liệu, cấu hình, tài liệu, script chạy pipeline và output thí nghiệm.

## Cấu trúc thư mục

```text
spam-detection/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   └── PULL_REQUEST_TEMPLATE.md
│
├── configs/
│   ├── augmentation_plan.json
│   ├── baseline.yaml
│   ├── linear_svc.yaml
│   └── logistic_regression.yaml
│
├── data/
│   ├── raw/
│   │   └── spam.csv
│   └── custom/
│       └── Assassin_preprocessed.csv
│
├── docs/
│   ├── Plan.md
│   ├── QA_QC.md
│   ├── augmentation_plan.md
│   ├── baseline_vs_best_logistic_regression_report.md
│   ├── datacollection_guide.md
│   ├── feedback_preprocess.md
│   ├── folder_workflow.md
│   ├── logistic_regression_vs_linear_svc_report.md
│   └── structure.md
│
├── experiments/
│   ├── baseline/
│   ├── linear_svc/
│   ├── logistic_regression/
│   └── splits/
│       └── split_indices.json
│
├── notebooks/
│
├── scripts/
│   ├── analyze_errors.py
│   ├── augment_data.py
│   ├── create_split.py
│   ├── evaluate_baseline.py
│   ├── evaluate_logistic_regression.py
│   ├── run_linear_svc_experiments.py
│   ├── run_logistic_experiments.py
│   ├── streamlit_app.py
│   ├── train_baseline.py
│   └── train_logistic_regression.py
│
├── src/
│   ├── data/
│   │   ├── preprocess_fixed.py
│   │   └── validation.py
│   ├── evaluation/
│   │   └── metrics.py
│   ├── models/
│   └── utils/
│
├── tests/
│   ├── test_preprocess.py
│   └── test_split_integrity.py
│
├── README.md
└── requirements.txt
```

## Vai trò từng thư mục

| Thư mục | Vai trò | Quy ước |
| :--- | :--- | :--- |
| `.github/` | Template cho issue và pull request | Giữ nguyên, không trộn code hoặc tài liệu dự án vào đây |
| `configs/` | Cấu hình chạy pipeline và kế hoạch augmentation dạng máy đọc được |
| `data/` | Dữ liệu của dự án | CSV/ZIP/dataset đặt ở đây, không đặt trong `src/` |
| `docs/` | Tài liệu | Kế hoạch, hướng dẫn, báo cáo, cấu trúc repo |
| `experiments/` | Output của pipeline | Metrics, prediction, report, split index, artifact thí nghiệm |
| `notebooks/` | Khám phá dữ liệu và thử nghiệm nhanh | Không để production code trong notebook |
| `scripts/` | Entry point chạy từ CLI | Script đọc config, gọi code trong `src/`, ghi output ra `experiments/` hoặc artifact hợp lệ |
| `src/` | Library code importable | Chỉ chứa Python module dùng lại được, không chứa file data |
| `tests/` | Kiểm thử | Test cho preprocess, split và logic ổn định của repo |

## Nguyên tắc phân loại

- `src/` chỉ chứa code Python có thể import.
- `scripts/` chứa file chạy trực tiếp, ví dụ train, evaluate, analyze, Streamlit app.
- `data/` chứa dữ liệu: raw data trong `data/raw/`, dữ liệu custom hoặc đã xử lý trong `data/custom/`.
- `configs/` chứa cấu hình và file JSON được workflow cần đọc. Riêng `augmentation_plan.json` được giữ lại vì còn được sử dụng.
- `docs/` chứa tài liệu Markdown cho người đọc. `docs/augmentation_plan.md` là bản human-readable của kế hoạch augmentation.
- `experiments/` chứa output sinh ra từ pipeline, bao gồm metrics, error analysis, split index và kết quả chạy model.

## Các thay đổi đã áp dụng

| Trước | Sau | Lý do |
| :--- | :--- | :--- |
| `src/data/raw/spam.csv` | `data/raw/spam.csv` | Dataset gốc không nên nằm trong source code |
| `src/data/Assassin_preprocessed.csv` | `data/custom/Assassin_preprocessed.csv` | Dataset custom/preprocessed thuộc nhóm dữ liệu |
| `augment_data.py` | `scripts/augment_data.py` | Script thực thi thuộc `scripts/` |
| `streamlit_app.py` | `scripts/streamlit_app.py` | App chạy trực tiếp cũng là entry point |
| `datacollection_guide.md` | `docs/datacollection_guide.md` | Tài liệu thuộc `docs/` |
| `src/training/` | Xóa | Thư mục rỗng, trách nhiệm bị trùng với `scripts/`, `src/models/`, `src/utils/` |

## Lưu ý về augmentation plan

Ban đầu `configs/augmentation_plan.json` được xem như generated output, nhưng file này có thể đang được workflow hoặc người khác đọc trực tiếp. Vì vậy repo giữ lại file này trong `configs/`. Script `scripts/augment_data.py` tiếp tục sinh `configs/augmentation_plan.json`, đồng thời sinh thêm `docs/augmentation_plan.md` để đọc dễ hơn.
