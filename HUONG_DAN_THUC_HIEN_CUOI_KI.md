# Hướng dẫn làm bài cuối kì Machine Learning từ căn bản

File này giải thích theo kiểu người mới bắt đầu cũng có thể hiểu. Phần artifact cuối cùng vẫn sẽ viết bằng tiếng Anh theo yêu cầu đề, nhưng file hướng dẫn này viết bằng tiếng Việt.

---

## 1. Bài này thật ra đang bắt mình làm gì?

Bài yêu cầu xây một hệ thống dự đoán học bổng.

Mỗi dòng dữ liệu là một sinh viên. Mỗi sinh viên có các thông tin như:

- GPA
- Attendance rate
- Study hours per week
- Exam score
- Household income
- Part time job hours

Máy phải học từ những sinh viên đã biết kết quả để dự đoán sinh viên mới có học bổng hay không.

Kết quả cần dự đoán là:

- `label = 1`: có học bổng.
- `label = 0`: không có học bổng.

Đây là bài toán classification vì đáp án là nhóm 0 hoặc nhóm 1.

---

## 2. Train, dev, hidden test là gì?

Train set:

- Là dữ liệu để model học.
- Trong bài này là `final/data/ml_dataset_v2_train.csv`.
- Có cả feature và label.

Dev set:

- Là dữ liệu để kiểm tra model sau khi học.
- Trong bài này là `final/data/ml_dataset_v2_dev.csv`.
- Có cả feature và label.
- Không nên dùng dev để train trực tiếp.

Hidden test:

- Là dữ liệu giáo viên dùng để chấm prediction.
- Thường hidden test không có label.
- Trong workspace hiện tại, file dùng để tạo prediction là `final/hidden_test_data.csv`.
- Notebook đọc file này bằng đường dẫn đơn giản `hidden_test_data.csv`, nghĩa là file cần nằm cùng folder với notebook khi chạy.

Quan trọng:

- Train để học.
- Dev để đánh giá và so sánh.
- Hidden test để tạo file nộp `predictions.csv`.

---

## 3. Feature và label là gì?

Feature là thông tin đầu vào.

Ví dụ:

- `gpa`
- `attendance_rate`
- `study_hours_per_week`
- `exam_score`
- `household_income`
- `part_time_job_hours`

Label là đáp án thật.

Trong bài này:

- `label` là đáp án.
- Không được dùng `label` như feature.

Ví dụ dễ hiểu:

- Nếu mình muốn đoán một học sinh có đậu không, mình dùng điểm chuyên cần, điểm thi, giờ học.
- Mình không được dùng cột "đậu hay rớt" làm dữ liệu đầu vào, vì đó chính là đáp án.

---

## 4. Vì sao phải bỏ cột id?

Cột `id` chỉ là mã số dòng hoặc mã sinh viên.

Nó không nói lên năng lực học tập thật.

Nếu đưa `id` vào model, model có thể học nhầm rằng số id lớn hay nhỏ liên quan đến học bổng. Đây là sai.

Vì vậy:

- Giữ `id` để xuất `predictions.csv`.
- Không dùng `id` để train model.

---

## 5. EDA là gì?

EDA là Exploratory Data Analysis.

Nói đơn giản, EDA là nhìn dữ liệu trước khi train.

Mục tiêu của EDA:

- Biết dataset có bao nhiêu dòng.
- Biết có bao nhiêu cột.
- Biết cột nào là feature, cột nào là label.
- Kiểm tra có missing value không.
- Kiểm tra label có bị lệch quá không.
- Nhìn chart để hiểu feature nào có vẻ quan trọng.

Ví dụ:

- Nếu GPA của nhóm có học bổng thường cao hơn nhóm không học bổng, thì GPA có vẻ là feature quan trọng.
- Nếu household income gần như không khác nhau giữa 2 nhóm, thì feature đó có thể ít quan trọng hơn.

---

## 6. Missing value là gì?

Missing value là ô dữ liệu bị trống.

Ví dụ:

| gpa | exam_score | label |
| --- | --- | --- |
| 3.5 | 80 | 1 |
| 2.8 | trống | 0 |

Nếu có missing value, ta phải xử lý trước khi train.

Trong dataset final hiện tại:

- Không có missing value.

Nhưng notebook vẫn cần kiểm tra để chứng minh với giáo viên là mình đã xem.

---

## 7. Scaling là gì và vì sao cần?

Các feature có thang đo khác nhau.

Ví dụ:

- GPA nằm khoảng 2.0 đến 4.0.
- Household income có thể từ vài trăm đến vài nghìn.

Nếu không scale, feature có số lớn như household income có thể làm model hiểu nhầm rằng nó quan trọng hơn chỉ vì giá trị số lớn.

Scaling biến dữ liệu về cùng thang đo.

Ví dụ Standard Scaling:

- Trừ đi trung bình.
- Chia cho độ lệch chuẩn.

Cực kì quan trọng:

- Fit scaler trên train.
- Dùng scaler đó transform dev và hidden.
- Không fit scaler trên toàn bộ dữ liệu vì sẽ gây data leakage.

---

## 8. Data leakage là gì?

Data leakage là khi model vô tình nhìn thấy thông tin mà đáng lẽ nó không được biết.

Ví dụ sai:

- Gộp train và dev lại.
- Fit scaler trên cả train và dev.
- Sau đó train model.

Vì sao sai?

- Dev set đáng lẽ dùng để kiểm tra như dữ liệu mới.
- Nếu scaler đã học từ dev, model gián tiếp biết thông tin của dev.
- Kết quả dev có thể đẹp giả tạo.

Cách đúng:

- Fit scaler trên train.
- Transform dev bằng thông số của train.

---

## 9. Baseline là gì?

Baseline là phiên bản đơn giản đầu tiên để so sánh.

Trong bài này baseline có thể là:

- Dùng feature gốc.
- Scaling cơ bản.
- Train Logistic Regression.
- Đánh giá Accuracy, Precision, Recall, F1.

Sau đó final phải cải tiến hơn baseline.

Ví dụ cải tiến:

- Thêm feature engineering.
- Tune hyperparameters.
- Dùng Random Forest.
- Tune threshold.
- Error analysis kỹ hơn.

---

## 10. Logistic Regression là gì?

Logistic Regression là model linear cho classification.

Nó tính một điểm số từ các feature.

Ý tưởng đơn giản:

```text
score = weight1 * gpa + weight2 * exam_score + ...
```

Sau đó đưa score qua sigmoid để ra xác suất từ 0 đến 1.

Nếu xác suất >= threshold:

- Dự đoán label 1.

Nếu xác suất < threshold:

- Dự đoán label 0.

Ưu điểm:

- Dễ giải thích.
- Coefficient cho biết feature nào đẩy prediction lên hoặc xuống.
- Phù hợp làm baseline.

Nhược điểm:

- Chỉ học quan hệ gần tuyến tính.
- Có thể yếu nếu dữ liệu có pattern phức tạp.

---

## 11. Decision Tree là gì?

Decision Tree giống như cây câu hỏi.

Ví dụ:

```text
Nếu gpa <= 3.1:
    dự đoán 0
Nếu gpa > 3.1:
    Nếu exam_score > 80:
        dự đoán 1
    Ngược lại:
        dự đoán 0
```

Model tự tìm câu hỏi tốt nhất để chia dữ liệu.

Ưu điểm:

- Dễ hiểu.
- Bắt được quan hệ phi tuyến.

Nhược điểm:

- Dễ overfit nếu cây quá sâu.

---

## 12. Random Forest là gì?

Random Forest là nhiều Decision Tree cộng lại.

Mỗi tree nhìn một phần dữ liệu và một phần feature.

Khi predict:

- Mỗi tree bỏ phiếu.
- Nhãn nào nhiều phiếu hơn thì chọn.

Ưu điểm:

- Ổn định hơn Decision Tree.
- Thường giảm overfitting.
- Có feature importance.

Nhược điểm:

- Khó giải thích hơn một cây đơn.
- Code dài hơn.

---

## 13. KNN là gì?

KNN là K-Nearest Neighbors.

Nó không học weight phức tạp.

Khi gặp một sinh viên mới:

- Tính khoảng cách đến các sinh viên trong train.
- Lấy `k` sinh viên gần nhất.
- Xem trong `k` người đó, label nào nhiều hơn.
- Dự đoán theo label nhiều hơn.

Ví dụ:

- k = 5.
- Trong 5 người gần nhất có 3 người label 1, 2 người label 0.
- KNN dự đoán label 1.

Ưu điểm:

- Dễ hiểu.
- Dễ tự code.

Nhược điểm:

- Cần scaling.
- Chạy chậm nếu dữ liệu lớn.

---

## 14. Feature engineering là gì?

Feature engineering là tạo thêm feature mới từ feature cũ.

Mục tiêu:

- Giúp model nhìn thấy pattern rõ hơn.

Ví dụ trong bài này:

```text
academic_strength = gpa * exam_score
```

Nghĩa là sinh viên vừa GPA cao vừa exam score cao thì academic strength cao.

Feature khác:

```text
study_efficiency = exam_score / (study_hours_per_week + 1)
```

Nghĩa là mỗi giờ học tạo ra bao nhiêu điểm thi.

Lưu ý:

- Không được dùng label để tạo feature.
- Không được tạo feature bằng đáp án.

---

## 15. Hyperparameter tuning là gì?

Hyperparameter là thông số mình chọn trước khi train.

Ví dụ:

- Logistic Regression: learning rate, số vòng lặp.
- Decision Tree: max depth.
- Random Forest: số cây.
- KNN: số neighbor `k`.

Tuning là thử vài giá trị rồi chọn giá trị tốt.

Ví dụ:

- KNN thử k = 3, 5, 7, 9.
- Cái nào F1 cao nhất thì chọn.

Không nên thử quá nhiều một cách mù quáng.

Phải giải thích vì sao chọn.

---

## 16. Accuracy, Precision, Recall, F1 là gì?

Accuracy:

- Tỉ lệ dự đoán đúng trên toàn bộ dữ liệu.
- Dễ hiểu nhưng có thể gây hiểu nhầm nếu data lệch lớp.

Precision:

- Trong những người model đoán có học bổng, bao nhiêu người thật sự có học bổng.
- Cao nghĩa là ít false positive.

Recall:

- Trong những người thật sự có học bổng, model tìm ra được bao nhiêu.
- Cao nghĩa là ít false negative.

F1-score:

- Trung bình hài hòa giữa precision và recall.
- Dùng tốt khi cần cân bằng.

Trong bài học bổng:

- False negative là người xứng đáng có học bổng nhưng model bỏ sót.
- False positive là người không có học bổng nhưng model dự đoán có.

Nếu ưu tiên công bằng cho sinh viên đủ điều kiện:

- Recall quan trọng.

Nếu ưu tiên tránh trao học bổng sai:

- Precision quan trọng.

---

## 17. Confusion matrix là gì?

Confusion matrix là bảng đếm đúng sai.

Với binary classification:

| | Predicted 0 | Predicted 1 |
| --- | --- | --- |
| Actual 0 | True Negative | False Positive |
| Actual 1 | False Negative | True Positive |

Ý nghĩa:

- True Positive: đoán có học bổng và đúng.
- True Negative: đoán không có học bổng và đúng.
- False Positive: đoán có học bổng nhưng sai.
- False Negative: đoán không có học bổng nhưng sai.

Report phải có confusion matrix.

---

## 18. Error analysis là làm gì?

Error analysis là nhìn vào những dòng model dự đoán sai.

Cần trả lời:

- Model sai ở false positive hay false negative nhiều hơn?
- Những dòng sai có điểm gì giống nhau?
- Có phải GPA ở mức lưng chừng không?
- Có phải exam score và GPA mâu thuẫn nhau không?
- Có phải part time job hours làm model khó đoán hơn không?

Ví dụ cách viết:

- "Most false positives are students with moderate GPA but high exam scores."
- "False negatives tend to have lower study hours but still receive scholarships, suggesting missing contextual factors."

Không nên chỉ nói:

- "The model made 3 errors."

Phải giải thích lỗi có pattern gì.

---

## 19. Model interpretation là gì?

Model interpretation là giải thích model dựa vào feature nào.

Ví dụ Logistic Regression:

- Coefficient dương lớn: feature đó làm tăng khả năng dự đoán label 1.
- Coefficient âm lớn: feature đó làm giảm khả năng dự đoán label 1.

Ví dụ Random Forest:

- Feature importance cao: feature đó được dùng nhiều để chia dữ liệu.

Lưu ý cực kì quan trọng:

- Feature quan trọng không có nghĩa là nguyên nhân thật.
- Chỉ được nói "associated with prediction", không nói "causes scholarship".

---

## 20. Vì sao final phải khác midterm?

Midterm chỉ cần pipeline cơ bản.

Final yêu cầu:

- Rõ baseline.
- So sánh ít nhất 3 model.
- Có cải tiến.
- Có model interpretation.
- Có error analysis sâu hơn.
- Report chuyên nghiệp hơn.

Vì vậy final không nên chỉ copy midterm.

Final phải nói:

- Midterm đã làm gì.
- Final cải tiến gì.
- Cải tiến đó làm metric hoặc phân tích tốt hơn như thế nào.

---

## 21. Cách tạo predictions.csv

File `predictions.csv` phải có đúng format:

```csv
id,label_pred
1001,1
1002,0
1003,0
```

Không được thêm cột khác.

Không được thêm index.

`label_pred` chỉ được là:

- 0
- 1

Hiện tại hidden test nằm ở:

```text
final/hidden_test_data.csv
```

Nếu file hidden test thiếu cột:

```text
part_time_job_hours
```

Cách xử lý:

- Lấy median của `part_time_job_hours` trong train final.
- Thêm cột này vào hidden test trước khi predict.
- Nếu hidden test mới đã có đủ cột thì bước điền median này không làm thay đổi gì không cần thiết.

---

## 22. Quy trình chạy bài từ đầu đến cuối

Bước 1:

- Mở folder project.

Bước 2:

- Kiểm tra có đủ data:
  - `final/ml_dataset_v2_train.csv`
  - `final/ml_dataset_v2_dev.csv`
  - `final/hidden_test_data.csv`

Bước 3:

- Mở `final/notebook.ipynb`.

Bước 4:

- Run all cells.

Bước 5:

- Kiểm tra output:
  - Model comparison table.
  - Confusion matrix.
  - Error analysis.
  - Feature importance.
  - `final/predictions.csv`.

Bước 6:

- Mở `final/report.pdf`.

Bước 7:

- Kiểm tra report đủ ít nhất 25 trang.

Bước 8:

- Nộp:
  - `report.pdf`
  - `notebook.ipynb`
  - `predictions.csv`

Trong workspace này, 3 file trên đã được gom riêng vào folder `submission/`.

---

## 23. Nếu giáo viên đưa hidden test final mới thì làm gì?

Nếu có file hidden test final mới:

1. Đặt file đó vào `final/data/`.
2. Mở notebook.
3. Sửa đường dẫn hidden test.
4. Run lại notebook.
5. Kiểm tra `final/predictions.csv` mới.

Nếu hidden test final có đủ cột:

- Không cần thêm `part_time_job_hours` bằng median.

Nếu hidden test final không có label:

- Vẫn tạo prediction bình thường.
- Không tính hidden accuracy vì không có đáp án.

---

## 24. Những lỗi dễ mất điểm

Lỗi 1: Dùng scikit-learn để train model.

- Không được làm vì constraint của bài mình là tự code thuật toán.

Lỗi 2: Đưa `id` vào feature.

- Dễ làm model học sai.

Lỗi 3: Fit scaler trên train + dev + hidden.

- Đây là data leakage.

Lỗi 4: Chỉ ghi metric mà không giải thích.

- Report sẽ bị đánh giá nông.

Lỗi 5: Không có error analysis.

- Đề nói rõ đây là bắt buộc.

Lỗi 6: Không có model interpretation.

- Đề nói rõ cần có.

Lỗi 7: Report dưới 25 trang.

- Trang bìa ghi minimum 25 pages, nên làm 25 trang trở lên.

Lỗi 8: `predictions.csv` sai header.

- Header phải là `id,label_pred`.

---

## 25. Checklist đơn giản nhất

Nếu chỉ nhớ một danh sách, nhớ danh sách này:

- [ ] Đọc đề.
- [ ] Dùng final train/dev.
- [ ] Dùng `final/hidden_test_data.csv`.
- [ ] Không dùng scikit-learn để train.
- [ ] Bỏ `id` khỏi feature.
- [ ] Scale đúng cách.
- [ ] Train ít nhất 3 model.
- [ ] Có Logistic Regression.
- [ ] Có tree-based model.
- [ ] Có model thứ ba khác.
- [ ] Có cải tiến so với midterm.
- [ ] Có Accuracy, Precision, Recall, F1.
- [ ] Có Confusion Matrix.
- [ ] Có Error Analysis.
- [ ] Có Model Interpretation.
- [ ] Có Final Recommendation.
- [ ] Tạo `report.pdf`.
- [ ] Tạo `notebook.ipynb`.
- [ ] Tạo `predictions.csv`.
- [ ] Kiểm tra folder `submission/` chỉ có 3 file bắt buộc.

---

## 26. Kiểm tra bản nộp hiện tại

Folder nộp hiện tại là:

```text
submission/
```

Trong folder này chỉ nên có 3 file:

```text
report.pdf
notebook.ipynb
predictions.csv
```

Kết quả kiểm tra gần nhất:

- `report.pdf`: có 32 trang, compile từ LaTeX.
- `notebook.ipynb`: chạy full được, 17 code cells không lỗi.
- `predictions.csv`: đúng header `id,label_pred`, có 100 dòng.
- `label_pred`: chỉ gồm 0 và 1.
- Notebook không import `sklearn`.
- Notebook dùng path đơn giản, các CSV nằm cùng folder với notebook.

Để kiểm tra lại bằng tay:

1. Mở terminal ở folder gốc project.
2. Chạy notebook hoặc chạy script build:

```powershell
python final\build_final_artifacts.py
```

3. Nếu vừa sửa report LaTeX, compile lại:

```powershell
cd final
& 'C:\Users\meow\.codex\plugins\cache\openai-bundled\latex-tectonic\0.1.0\bin\tectonic.exe' --outdir . main.tex
Copy-Item -LiteralPath 'main.pdf' -Destination 'report.pdf' -Force
```

4. Copy lại 3 file mới nhất vào `submission/`.

Thông tin nhóm trong bản nộp hiện tại:

- Chau Pham Tuan Kiet - 523K0011
- Chau Thanh Vu - 523K0032
- Tran Anh Minh - 523K0017

Contribution statement đã có trong report:

- Chau Pham Tuan Kiet: phần nhẹ nhất, gồm kiểm tra file nộp, đường dẫn dữ liệu, format `predictions.csv`, notebook cleanup và reproducibility checks.
- Chau Thanh Vu: preprocessing, feature engineering, scaling, Logistic Regression, KNN và tuning.
- Tran Anh Minh: EDA, tree-based models, model comparison, error analysis, interpretation và chỉnh report.

Nếu sau này sửa `final/main.tex`, nhớ compile lại PDF rồi copy lại sang `submission/report.pdf`.

---

## 27. Code chi tiết để hiểu notebook

Phần này không thay thế `final/notebook.ipynb`. Nó chỉ giải thích các đoạn code quan trọng nhất để người mới hiểu bài đang chạy như thế nào.

### 27.1. Import thư viện và khai báo feature

Notebook dùng `numpy`, `pandas`, `matplotlib`, `seaborn`. Không import `sklearn`.

```python
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

MIDTERM_BASE_FEATURES = [
    "gpa",
    "attendance_rate",
    "study_hours_per_week",
    "exam_score",
    "household_income",
]

FINAL_BASE_FEATURES = MIDTERM_BASE_FEATURES + ["part_time_job_hours"]

ENGINEERED_FEATURES = [
    "academic_strength",
    "study_efficiency",
    "income_per_study_hour",
    "work_study_balance",
]

IMPROVED_FEATURES = FINAL_BASE_FEATURES + ENGINEERED_FEATURES
```

Ý nghĩa:

- `MIDTERM_BASE_FEATURES`: các feature cũ từ bài giữa kì.
- `FINAL_BASE_FEATURES`: thêm `part_time_job_hours`.
- `ENGINEERED_FEATURES`: feature tự tạo để cải tiến final.
- `IMPROVED_FEATURES`: toàn bộ feature dùng cho model final.

### 27.2. Đường dẫn đơn giản giống giữa kì

Các CSV được đặt cùng folder với notebook:

```python
PROJECT_DIR = Path.cwd()
FINAL_DIR = PROJECT_DIR

TRAIN_PATH = "ml_dataset_v2_train.csv"
DEV_PATH = "ml_dataset_v2_dev.csv"
HIDDEN_PATH = "hidden_test_data.csv"
```

Khi chạy trên Google Colab, upload chung các file này vào cùng thư mục:

```text
notebook.ipynb
ml_dataset_v2_train.csv
ml_dataset_v2_dev.csv
hidden_test_data.csv
```

### 27.3. Load dữ liệu

```python
train_df = pd.read_csv(TRAIN_PATH)
dev_df = pd.read_csv(DEV_PATH)
hidden_df = pd.read_csv(HIDDEN_PATH)

print("Train shape:", train_df.shape)
print("Dev shape:", dev_df.shape)
print("Hidden test shape:", hidden_df.shape)
display(train_df.head())
```

Kết quả mong muốn:

- Train: 250 dòng, 8 cột.
- Dev: 100 dòng, 8 cột.
- Hidden: 100 dòng.

### 27.4. Kiểm tra cột bắt buộc

Hàm này giúp phát hiện thiếu cột sớm:

```python
def validate_columns(df: pd.DataFrame, required_cols: list[str], name: str) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")
```

Cách dùng:

```python
validate_columns(
    train_df,
    ["id"] + FINAL_BASE_FEATURES + ["label"],
    "final train",
)

validate_columns(
    dev_df,
    ["id"] + FINAL_BASE_FEATURES + ["label"],
    "final dev",
)

validate_columns(
    hidden_df,
    ["id"] + MIDTERM_BASE_FEATURES,
    "hidden test",
)
```

Với hidden test, chỉ bắt buộc `MIDTERM_BASE_FEATURES` vì file hidden có thể thiếu `part_time_job_hours`.

### 27.5. Lấy median từ train

Nếu hidden test thiếu `part_time_job_hours`, notebook dùng median của train để điền.

```python
def feature_medians_from_train(train_df: pd.DataFrame) -> dict[str, float]:
    return {
        col: float(train_df[col].median())
        for col in FINAL_BASE_FEATURES
        if col in train_df.columns
    }

medians = feature_medians_from_train(train_df)
print(medians)
```

Trong bản hiện tại:

```text
part_time_job_hours median = 13.0
```

### 27.6. Tạo feature frame và xử lý hidden thiếu cột

Đây là phần quan trọng để train/dev/hidden có cùng schema feature.

```python
def build_feature_frame(
    df: pd.DataFrame,
    medians: dict[str, float],
    include_engineered: bool = True,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    working = df.copy()

    for col in FINAL_BASE_FEATURES:
        if col not in working.columns:
            working[col] = medians[col]

    if include_engineered:
        working["academic_strength"] = working["gpa"] * working["exam_score"]
        working["study_efficiency"] = (
            working["exam_score"] / (working["study_hours_per_week"] + 1.0)
        )
        working["income_per_study_hour"] = (
            working["household_income"] / (working["study_hours_per_week"] + 1.0)
        )
        working["work_study_balance"] = (
            working["study_hours_per_week"] - working["part_time_job_hours"]
        )

    if feature_cols is None:
        feature_cols = IMPROVED_FEATURES if include_engineered else FINAL_BASE_FEATURES

    return working[feature_cols].copy()
```

Ý nghĩa các feature tự tạo:

- `academic_strength`: kết hợp GPA và exam score.
- `study_efficiency`: điểm thi trên mỗi giờ học.
- `income_per_study_hour`: thu nhập gia đình so với giờ học.
- `work_study_balance`: cân bằng giữa giờ học và giờ làm thêm.

### 27.7. Standard scaler tự code

Scaler chỉ fit trên train, sau đó transform dev và hidden.

```python
class StandardScalerModel:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "StandardScalerModel":
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler has not been fitted.")
        return (x - self.mean_) / self.std_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.transform(x)
```

Cách dùng đúng:

```python
x_train = build_feature_frame(train_df, medians)
x_dev = build_feature_frame(dev_df, medians)
x_hidden = build_feature_frame(hidden_df, medians)

scaler = StandardScalerModel()
x_train_scaled = scaler.fit_transform(x_train.to_numpy(dtype=float))
x_dev_scaled = scaler.transform(x_dev.to_numpy(dtype=float))
x_hidden_scaled = scaler.transform(x_hidden.to_numpy(dtype=float))
```

Không được fit scaler trên dev hoặc hidden vì sẽ gây data leakage.

### 27.8. Hàm sigmoid cho Logistic Regression

```python
def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))
```

`np.clip` giúp tránh overflow khi số quá lớn hoặc quá nhỏ.

### 27.9. Logistic Regression tự code bằng NumPy

Đây là phiên bản rút gọn để hiểu ý tưởng:

```python
class LogisticRegressionModel:
    def __init__(
        self,
        learning_rate: float = 0.03,
        n_iterations: int = 5000,
        l2: float = 0.0,
    ) -> None:
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.l2 = l2
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LogisticRegressionModel":
        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iterations):
            scores = x @ self.weights + self.bias
            probs = sigmoid(scores)

            error = probs - y
            dw = (x.T @ error) / n_samples + self.l2 * self.weights
            db = float(error.mean())

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Model has not been fitted.")
        return sigmoid(x @ self.weights + self.bias)

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(x) >= threshold).astype(int)
```

Trong bản hiện tại, Logistic Regression được chọn với:

```text
learning_rate = 0.03
n_iterations = 5000
l2 = 0.0
threshold = 0.55
```

### 27.10. Metrics tự tính

```python
def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    counts = confusion_counts(y_true, y_pred)
    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        **counts,
    }
```

Với selected model trên dev:

```text
TP = 32
TN = 60
FP = 0
FN = 8
Accuracy = 0.9200
Precision = 1.0000
Recall = 0.8000
F1 = 0.8889
```

### 27.11. Tạo `predictions.csv`

Sau khi train selected model, tạo file nộp:

```python
hidden_scores = selected_model.predict_proba(x_hidden_scaled)
hidden_preds = (hidden_scores >= selected_threshold).astype(int)

submission = pd.DataFrame({
    "id": hidden_df["id"].astype(int),
    "label_pred": hidden_preds.astype(int),
})

submission.to_csv(FINAL_DIR / "predictions.csv", index=False)
display(submission.head())
```

Điểm phải nhớ:

- Không thêm index.
- Không thêm cột `score`.
- Không thêm cột `label`.
- Header phải đúng chính xác là `id,label_pred`.

### 27.12. Code kiểm tra `predictions.csv`

```python
pred = pd.read_csv("predictions.csv")

assert list(pred.columns) == ["id", "label_pred"]
assert pred["label_pred"].isin([0, 1]).all()
assert pred["id"].duplicated().sum() == 0
assert pred.isna().sum().sum() == 0

print(pred.shape)
print(pred["label_pred"].value_counts().sort_index())
```

Kết quả hiện tại:

```text
shape = (100, 2)
label_pred distribution = {0: 64, 1: 36}
```

### 27.13. Code chạy notebook bằng terminal để kiểm tra

Nếu muốn kiểm tra notebook chạy hết không lỗi:

```powershell
@'
from pathlib import Path
import nbformat
from nbclient import NotebookClient

final = Path.cwd() / "final"
nb_path = final / "notebook.ipynb"

nb = nbformat.read(nb_path, as_version=4)
client = NotebookClient(
    nb,
    timeout=900,
    kernel_name="python3",
    resources={"metadata": {"path": str(final)}},
)

client.execute()
print("NOTEBOOK_EXECUTION_OK")
'@ | python -
```

Nếu in ra:

```text
NOTEBOOK_EXECUTION_OK
```

nghĩa là notebook chạy từ đầu đến cuối không lỗi.

### 27.14. Code kiểm tra PDF và folder nộp

```powershell
@'
from pathlib import Path
import pandas as pd
from pypdf import PdfReader

sub = Path("submission")
files = sorted(p.name for p in sub.iterdir() if p.is_file())
print(files)

assert files == ["notebook.ipynb", "predictions.csv", "report.pdf"]

reader = PdfReader(str(sub / "report.pdf"))
print("Report pages:", len(reader.pages))
assert len(reader.pages) >= 25

pred = pd.read_csv(sub / "predictions.csv")
assert list(pred.columns) == ["id", "label_pred"]
assert pred["label_pred"].isin([0, 1]).all()
assert pred.isna().sum().sum() == 0
assert pred["id"].duplicated().sum() == 0

print("SUBMISSION_CHECK_OK")
'@ | python -
```

Nếu in ra:

```text
SUBMISSION_CHECK_OK
```

nghĩa là folder nộp đang đúng 3 file, PDF đủ trang, prediction đúng format.
