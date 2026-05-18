# Quy trình Làm việc với Cấu trúc Thư mục

> **Tham chiếu:** [structure.md](../structure.md) — mô tả cấu trúc hiện tại và đề xuất cấu trúc mới.

---

## Nguyên tắc Cốt lõi

| Thư mục | Vai trò | Được phép | Không được phép |
| :--- | :--- | :--- | :--- |
| `src/` | Library code | Viết hàm Python có thể import | Đọc file data trực tiếp, chứa file `.csv`/`.zip` |
| `scripts/` | Entry point CLI | Đọc config, gọi `src/`, ghi `experiments/` | Chứa business logic (phải đưa vào `src/`) |
| `data/` | Dữ liệu | Read-only sau khi đặt vào | Bị script ghi đè hoặc chỉnh sửa |
| `configs/` | Config tĩnh | File YAML/JSON định nghĩa hyperparameters | Chứa output được sinh ra bởi script |
| `experiments/` | Output pipeline | Script ghi kết quả vào đây | Commit file lớn (model `.joblib` nên gitignore) |
| `docs/` | Tài liệu | File Markdown: kế hoạch, hướng dẫn, báo cáo | Code Python, file data |
| `tests/` | Kiểm thử | Test cho code trong `src/` | Test phụ thuộc vào file trong `experiments/` |
| `notebooks/` | Exploration | EDA, prototyping, thử nghiệm nhanh | Code production (phải chuyển sang `src/`) |

---

## Quy trình Theo Từng Tình huống

### Thêm thuật toán mới (VD: SVM, Logistic Regression)

**Thứ tự bắt buộc:**

1. **Tạo model class trong `src/models/`**
   ```
   src/models/
   ├── base.py          # Abstract BaseClassifier (fit, predict, save, load)
   ├── naive_bayes.py
   └── svm.py           ← tạo mới
   ```
   - Không đọc file, không đọc config trực tiếp trong class
   - Nhận hyperparameters qua constructor

2. **Tạo config trong `configs/`**
   ```
   configs/
   ├── baseline.yaml    # NB — giữ nguyên
   └── svm.yaml         ← tạo mới, định nghĩa kernel/C/output path
   ```

3. **Tạo script trong `scripts/`**
   ```
   scripts/
   ├── train_baseline.py
   └── train_svm.py     ← tạo mới, chỉ orchestrate
   ```
   Script chỉ làm: đọc config → load data → gọi `src/` → ghi `experiments/`

4. **Output vào `experiments/` theo tên model**
   ```
   experiments/
   ├── baseline/
   └── svm/             ← tự động tạo khi script chạy
       ├── model.joblib
       └── metrics.json
   ```

5. **Viết test trong `tests/`**
   ```
   tests/
   └── test_svm.py      ← test class trong src/models/svm.py
   ```

---

### Thêm dữ liệu mới

1. **Dữ liệu gốc / chưa xử lý** → `data/raw/`
   ```
   data/raw/
   ├── spam.csv         # dataset gốc UCI — không chỉnh sửa
   └── new_dataset.csv  ← đặt vào đây
   ```

2. **Dữ liệu custom / đã làm sạch** → `data/custom/`
   ```
   data/custom/
   ├── Assassin_cleaned.csv
   └── new_custom_v2.csv
   ```

3. **Cập nhật config** nếu thay đổi dataset đầu vào:
   ```yaml
   # configs/baseline.yaml
   data:
     input_path: "data/raw/new_dataset.csv"
   ```

4. **Xóa split cũ** nếu thay đổi dataset (split không còn hợp lệ):
   ```bash
   rm experiments/splits/split_indices.json
   python scripts/create_split.py
   ```

> **Quy tắc cứng:** `data/` là read-only sau khi file được đặt vào. Script không bao giờ ghi vào `data/`.

---

### Thêm bước tiền xử lý mới

1. **Viết hàm vào `src/data/preprocess_fixed.py`** — không tạo file mới
2. **Thêm flag vào `configs/baseline.yaml`** (section `preprocess:`)
3. **Tất cả script và notebook** import từ `src.data.preprocess_fixed` — không duplicate logic

```python
# Đúng
from src.data.preprocess_fixed import preprocess_pipeline, clean_text

# Sai — hard-code logic tiền xử lý inline
text = text.lower().strip().replace(...)
```

---

### Viết tài liệu mới

| Loại tài liệu | Đặt ở đâu |
| :--- | :--- |
| Kế hoạch, roadmap | `docs/Plan.md` |
| Tiêu chí nghiệm thu | `docs/QA_QC.md` |
| Hướng dẫn thu thập data | `docs/datacollection_guide.md` |
| Phản hồi / quyết định kỹ thuật | `docs/feedback_*.md` |
| Quy trình làm việc | `docs/folder_workflow.md` (file này) |
| Báo cáo output của pipeline | `experiments/*/error_analysis/*.md` |
| Ghi chú tạm / scratchpad | `notebooks/` |

> **Không đặt tài liệu ở root.** Root chỉ chứa: `CLAUDE.md`, `README.md`, `requirements.txt`, `.gitignore`.

---

### Chạy pipeline đầy đủ

```bash
# Lần đầu (hoặc khi thay dataset):
python scripts/create_split.py

# Mỗi lần train lại:
python scripts/train_baseline.py
python scripts/evaluate_baseline.py
python scripts/analyze_errors.py
```

Không chạy `evaluate_baseline.py` trước `train_baseline.py` — file `model.joblib` chưa tồn tại.

---

## Checklist Trước Khi Commit

- [ ] Không có file `.csv` / `.zip` / `.joblib` mới trong `src/`
- [ ] Không có script thực thi mới nằm ở root (phải ở `scripts/`)
- [ ] Không có tài liệu `.md` mới nằm ở root (phải ở `docs/`)
- [ ] Config mới nằm trong `configs/`, không phải `experiments/`
- [ ] Logic tiền xử lý không bị duplicate — chỉ có 1 nguồn: `src/data/preprocess_fixed.py`
- [ ] `experiments/` không commit file model lớn (kiểm tra `.gitignore`)
