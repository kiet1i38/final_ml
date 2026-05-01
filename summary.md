# Summary Workspace Final ML

File này tóm tắt toàn bộ cuộc trò chuyện và trạng thái hiện tại của folder `D:\kiet\hk6\machine_learning\final_ml\final_ml_00`. Mục tiêu là để mở một cuộc chat mới, đưa file này cho assistant đọc, thì assistant hiểu được bài này đang làm gì, file nào quan trọng, đã chốt quyết định gì, và nên tiếp tục như thế nào.

## 1. Bối cảnh bài

Đây là bài cuối kì môn Introduction to Machine Learning.

Đề chính:

- `final_handout.pdf`
- `ml_project_student_guide.pdf`

Bài toán:

- Dự đoán sinh viên có nhận học bổng hay không.
- Đây là bài toán binary classification.
- Target là `label`.
- `label = 1`: nhận học bổng.
- `label = 0`: không nhận học bổng.

Theo đề, file bắt buộc phải nộp:

- `report.pdf`
- `notebook.ipynb`
- `predictions.csv`

File khuyến nghị:

- `README.md`
- `requirements.txt`

Report:

- Viết bằng tiếng Anh.
- Có figure, table, technical discussion.
- Có yêu cầu mâu thuẫn 15 trang và 25 trang; đã chọn mức an toàn là trên 25 trang.
- Bản hiện tại có 31 trang.

## 2. Yêu cầu và quyết định đã chốt trong cuộc trò chuyện

Người dùng yêu cầu:

- Tạo 2 file `.md` ở folder gốc:
  - một file kế hoạch cực kì chi tiết.
  - một file hướng dẫn từ căn bản cho người mới.
- Hai file `.md` này viết bằng tiếng Việt.
- Khi implement notebook/report thì viết bằng tiếng Anh.
- Sau khi lập kế hoạch thì làm toàn bộ artifact cuối kì luôn.
- Không dùng thư viện train ML như scikit-learn.
- Có thể dùng `pandas`, `numpy`, `matplotlib`, `seaborn`, `reportlab`, `nbformat`.
- Các thuật toán ML được tự code bằng NumPy.
- Model nên dễ làm, dễ giải thích, và đủ yêu cầu đề.
- Report phải vừa điểm cao vừa dễ hiểu/dễ bảo vệ.
- Dùng `midterm done` làm baseline/tham chiếu vì đó là folder bài giữa kì đã làm.
- Tạm thời dùng hidden test trong `midterm done` cho `predictions.csv` cho đến khi có hidden test final thật.
- Notebook phải chia cell nhỏ, dễ nhìn, dễ sửa.
- Notebook phải chạy được trên Google Colab.
- Sau đó người dùng muốn notebook tìm file đơn giản giống giữa kì: CSV nằm cùng folder với notebook.
- Report phải trình bày giống giữa kì vì giữa kì làm bằng LaTeX.
- Do đó đã tạo `final/main.tex` và compile thành `final/report.pdf`.
- Người dùng không muốn chữ hiển thị `from scratch`; đã bỏ cụm này khỏi notebook, report, README, PDF text.

## 3. Cấu trúc folder gốc hiện tại

Folder gốc hiện có:

- `final/`
- `midterm/`
- `midterm done/`
- `final_handout.pdf`
- `ml_project_student_guide.pdf`
- `KE_HOACH_CUOI_KI.md`
- `HUONG_DAN_THUC_HIEN_CUOI_KI.md`
- `summary.md`

Hai file kế hoạch/hướng dẫn tiếng Việt đã tạo:

- `KE_HOACH_CUOI_KI.md`
- `HUONG_DAN_THUC_HIEN_CUOI_KI.md`

## 4. Folder `midterm done`

Folder `midterm done/` là bài giữa kì đã làm xong, dùng để tham chiếu baseline.

Nó có các file quan trọng:

- `notebook.ipynb`
- `report.pdf`
- `report.docx`
- `report.md`
- `predictions.csv`
- `hidden_test_data.csv`
- `ml_dataset_v1_train.csv`
- `ml_dataset_v1_dev.csv`
- tài liệu giữa kì như handout, grading policy, submission rules.

Lưu ý:

- `midterm done/report.md` vẫn có placeholder `Team Name: [Your Team Names]`.
- Không tìm thấy tên nhóm/thành viên thật trong report giữa kì.
- Vì vậy report final hiện dùng `Team Name: Student Team`.
- Nếu cần nộp thật, người dùng nên thay tên nhóm/thành viên/MSSV trong `final/main.tex`.

## 5. Folder `final` hiện tại

Folder `final/` hiện là folder chính để nộp.

Các file quan trọng trong `final/`:

- `main.tex`: source LaTeX của report.
- `report.pdf`: PDF report compile từ `main.tex`.
- `main.pdf`: PDF gốc do Tectonic compile từ `main.tex`, sau đó copy thành `report.pdf`.
- `notebook.ipynb`: notebook cuối kì đã chia cell nhỏ.
- `predictions.csv`: file dự đoán nộp.
- `README.md`: hướng dẫn chạy.
- `requirements.txt`: thư viện cần cài.
- `build_final_artifacts.py`: script tạo lại artifact/model/figure/result/notebook.
- `ml_dataset_v2_train.csv`: final train CSV đặt cùng folder notebook.
- `ml_dataset_v2_dev.csv`: final dev CSV đặt cùng folder notebook.
- `hidden_test_data.csv`: hidden test tạm copy từ `midterm done`.

Các folder con trong `final/`:

- `data/`: vẫn giữ dataset final gốc.
- `figures/`: chứa hình dùng trong report.
- `results/`: chứa CSV/JSON kết quả model.
- `docs/`, `handout/`, `notebook/`, `report/`, `submission_format/`: một số folder template/empty từ bộ đề ban đầu.

## 6. Dataset final

Final train:

- File cùng folder notebook: `final/ml_dataset_v2_train.csv`
- File gốc: `final/data/ml_dataset_v2_train.csv`
- Shape: `(250, 8)`

Final dev:

- File cùng folder notebook: `final/ml_dataset_v2_dev.csv`
- File gốc: `final/data/ml_dataset_v2_dev.csv`
- Shape: `(100, 8)`

Hidden test tạm:

- File cùng folder notebook: `final/hidden_test_data.csv`
- Nguồn copy từ: `midterm done/hidden_test_data.csv`
- Shape: `(100, 7)`
- File này thiếu `part_time_job_hours` vì nó là schema giữa kì.
- Notebook xử lý bằng cách thêm cột thiếu bằng median của final train.

Các cột final:

- `id`
- `gpa`
- `attendance_rate`
- `study_hours_per_week`
- `exam_score`
- `household_income`
- `part_time_job_hours`
- `label`

Target:

- `label`

Feature final:

- bỏ `id`.
- bỏ `label`.
- dùng các cột feature còn lại.

## 7. Notebook cuối kì

File:

- `final/notebook.ipynb`

Trạng thái:

- 35 cells.
- Đã chia nhỏ, không còn một cell khổng lồ.
- Output đã được dọn sạch để đem lên Colab không chứa path máy local.
- Không import sklearn.
- Không có absolute Windows path.
- File path đơn giản như giữa kì.

Phần path hiện tại trong notebook:

```python
PROJECT_DIR = Path.cwd()
FINAL_DIR = PROJECT_DIR

TRAIN_PATH = 'ml_dataset_v2_train.csv'
DEV_PATH = 'ml_dataset_v2_dev.csv'
HIDDEN_PATH = 'hidden_test_data.csv'
```

Vì vậy khi chạy notebook, cần đặt các file này cùng folder với notebook:

- `notebook.ipynb`
- `ml_dataset_v2_train.csv`
- `ml_dataset_v2_dev.csv`
- `hidden_test_data.csv`

Trên Google Colab:

- upload 4 file trên vào cùng working folder.
- chạy notebook từ trên xuống dưới.

Các heading chính trong notebook:

- `Setup and Imports`
- `File Paths`
- `Load and Inspect Data`
- `Exploratory Data Analysis`
- `Data Validation and Feature Engineering Helpers`
- `Scaler`
- `Model 1: Logistic Regression`
- `Model 2: Decision Tree`
- `Model 3: Random Forest`
- `Model 4: KNN`
- `Metrics and Evaluation Helpers`
- `Hyperparameter Tuning Helpers`
- `Full Pipeline and Figure Generation`
- `Run Full Training Pipeline`
- `Error Analysis`
- `Model Interpretation`
- `Generate Submission Predictions`

Tên class model trong notebook:

- `StandardScalerModel`
- `LogisticRegressionModel`
- `DecisionTreeModel`
- `RandomForestModel`
- `KNNModel`

Người dùng không muốn chữ `from scratch`, nên không dùng cụm đó trong heading/tài liệu.

## 8. Model và kết quả hiện tại

Ràng buộc:

- Không dùng scikit-learn để train.
- Các model được viết bằng NumPy.

Model được train/so sánh:

- Logistic Regression
- Decision Tree
- Random Forest
- KNN

Model được chọn:

- Logistic Regression

Lý do:

- Logistic Regression và KNN hòa nhau ở Accuracy/Precision/Recall/F1.
- Logistic Regression có ROC-AUC cao hơn một chút.
- Logistic Regression dễ giải thích hơn bằng coefficients.

Kết quả trên final dev:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9200 | 1.0000 | 0.8000 | 0.8889 | 0.9963 | 0 | 8 |
| KNN | 0.9200 | 1.0000 | 0.8000 | 0.8889 | 0.9933 | 0 | 8 |
| Random Forest | 0.8900 | 0.9394 | 0.7750 | 0.8493 | 0.9458 | 2 | 9 |
| Decision Tree | 0.7900 | 0.7714 | 0.6750 | 0.7200 | 0.7613 | 8 | 13 |

Selected Logistic Regression params:

- `learning_rate = 0.03`
- `n_iterations = 5000`
- `l2 = 0.0`
- threshold: `0.55`

Dev confusion matrix selected model:

- TP = 32
- TN = 60
- FP = 0
- FN = 8

Temporary hidden local check:

- Accuracy: 0.8400
- Precision: 0.8333
- Recall: 0.7500
- F1: 0.7895
- Đây không phải kết quả hidden final chính thức.

## 9. Predictions file

File:

- `final/predictions.csv`

Format:

```csv
id,label_pred
```

Trạng thái:

- Shape: `(100, 2)`
- Columns: `['id', 'label_pred']`
- Label distribution:
  - 0: 64
  - 1: 36

File này dùng `final/hidden_test_data.csv`, vốn copy từ `midterm done/hidden_test_data.csv`.

## 10. Report LaTeX

File source:

- `final/main.tex`

PDF nộp:

- `final/report.pdf`

PDF compile output gốc:

- `final/main.pdf`

Tool compile đã dùng:

```powershell
& 'C:\Users\meow\.codex\plugins\cache\openai-bundled\latex-tectonic\0.1.0\bin\tectonic.exe' --outdir . main.tex
Copy-Item -LiteralPath 'main.pdf' -Destination 'report.pdf' -Force
```

Khi compile, Tectonic có cảnh báo:

- `Fontconfig error: Cannot load default config file`
- `warning: Object @page.1 already defined`

Nhưng exit code là 0 và PDF compile được. Không còn cảnh báo overfull/underfull quan trọng sau lần chỉnh cuối.

PDF hiện tại:

- `final/report.pdf`
- 31 pages
- Metadata: LaTeX/hyperref, xdvipdfmx.
- Đã kiểm tra text có title và các section chính.

Các section chính trong report:

- Introduction
- Dataset
- Exploratory Data Analysis
- Data Preprocessing
- Models
- Improvement Strategy
- Experimental Results
- Error Analysis
- Model Interpretation
- Final Recommendation and Conclusion
- Limitations
- Appendix

Lưu ý:

- Report không còn dùng ReportLab làm nguồn chính.
- `final/report.md` cũ đã bị xóa để tránh nhầm.
- Nếu sửa report, sửa `final/main.tex`, compile lại ra `main.pdf`, rồi copy thành `report.pdf`.

## 11. README và requirements

README:

- `final/README.md`

README ghi:

- file cần nộp.
- data dùng.
- cách chạy notebook.
- cách rebuild report từ LaTeX.
- cách chạy trên Google Colab.

Requirements:

- `final/requirements.txt`

Nội dung requirements:

```text
numpy
pandas
matplotlib
seaborn
reportlab
nbformat
```

Lưu ý:

- `reportlab` hiện không còn là nguồn chính của report, nhưng vẫn còn trong requirements vì `build_final_artifacts.py` vẫn có code fallback/report generator cũ và dependency này đã được dùng trước đó.
- Nếu muốn tối giản requirements sau này có thể cân nhắc bỏ `reportlab`, nhưng hiện giữ lại để không làm hỏng script.

## 12. Figures và results

Folder figures:

- `final/figures/`

Các hình quan trọng trong report:

- `target_distribution.png`
- `correlation_heatmap.png`
- `gpa_by_label.png`
- `part_time_by_label.png`
- `model_comparison_f1.png`
- `selected_confusion_matrix.png`
- `error_feature_difference.png`
- `permutation_importance.png`

Folder results:

- `final/results/`

Các file kết quả:

- `baseline_result.csv`
- `best_configs.csv`
- `dev_predictions_with_errors.csv`
- `error_cases.csv`
- `logistic_coefficients.csv`
- `model_comparison.csv`
- `permutation_importance.csv`
- `random_forest_importance.csv`
- `summary.json`
- `tuning_results.csv`

`final/results/summary.json` chứa:

- data summary.
- selected model.
- temporary hidden metrics.
- hidden missing feature note.

## 13. Script build artifact

File:

- `final/build_final_artifacts.py`

Vai trò:

- Chạy lại model pipeline.
- Sinh lại figures.
- Sinh lại results.
- Sinh lại `predictions.csv`.
- Sinh lại `notebook.ipynb`.
- Sinh lại `README.md` và `requirements.txt`.

Quan trọng:

- Script hiện không overwrite `report.pdf` bằng ReportLab nếu `main.tex` tồn tại.
- Khi chạy script, nó sẽ in:
  - `Skipped ReportLab report generation because final/main.tex exists.`
  - `Compile final/main.tex to rebuild report.pdf from LaTeX.`
- Nghĩa là report chính vẫn là LaTeX.

Cách chạy:

```powershell
python final\build_final_artifacts.py
```

Sau đó nếu cần rebuild report PDF:

```powershell
cd final
& 'C:\Users\meow\.codex\plugins\cache\openai-bundled\latex-tectonic\0.1.0\bin\tectonic.exe' --outdir . main.tex
Copy-Item -LiteralPath 'main.pdf' -Destination 'report.pdf' -Force
```

Nếu không dùng máy này, chỉ cần dùng TeX compiler tương đương để compile `main.tex`.

## 14. Các thay đổi lớn đã làm trong cuộc trò chuyện

Theo thứ tự:

1. Đọc folder ban đầu:
   - thấy `final/`, `midterm/`, `final_handout.pdf`, `ml_project_student_guide.pdf`.
2. Đọc đề cuối kì:
   - yêu cầu `report.pdf`, `notebook.ipynb`, `predictions.csv`.
   - report tiếng Anh, tối thiểu an toàn 25 trang.
3. Đọc dataset final:
   - train 250 dòng, dev 100 dòng.
   - final thêm `part_time_job_hours`.
4. Người dùng thêm `midterm done/`.
5. Đọc `midterm done`:
   - có notebook/report/predictions/hidden test giữa kì.
6. Tạo hai file tiếng Việt:
   - `KE_HOACH_CUOI_KI.md`
   - `HUONG_DAN_THUC_HIEN_CUOI_KI.md`
7. Tạo pipeline model và artifact final.
8. Tạo `final/report.pdf` ban đầu bằng ReportLab.
9. Người dùng muốn report giống giữa kì hơn:
   - chỉnh report style.
10. Người dùng nói giữa kì làm bằng LaTeX:
   - tạo `final/main.tex`.
   - compile thành `final/report.pdf`.
11. Người dùng muốn notebook dễ sửa:
   - chia notebook thành nhiều cell nhỏ.
12. Người dùng muốn notebook không dùng path phức tạp:
   - đặt CSV cùng folder `final/`.
   - notebook dùng `TRAIN_PATH`, `DEV_PATH`, `HIDDEN_PATH` như giữa kì.
13. Người dùng không muốn chữ `from scratch`:
   - đổi headings và class names.
   - xóa cụm `from scratch`, `from-scratch`, `Scratch` khỏi notebook/report/README/PDF text.
14. Người dùng yêu cầu tạo file summary:
   - file hiện tại là `summary.md`.

## 15. Những gì đã verify

Các kiểm tra đã chạy nhiều lần trong quá trình làm:

- Notebook chạy được bằng `nbclient` từ folder `final/`.
- Notebook không import sklearn.
- Notebook không có absolute Windows path.
- Notebook output đã clean.
- Notebook hiện có 35 cells.
- `predictions.csv` đúng columns `id,label_pred`.
- `predictions.csv` có 100 dòng.
- `label_pred` chỉ gồm 0 và 1.
- `report.pdf` compile từ LaTeX.
- `report.pdf` có 31 trang.
- `main.tex` tồn tại.
- `report.md` cũ đã bị xóa.
- Không còn cụm `from scratch`, `from-scratch`, `Scratch` trong:
  - `final/notebook.ipynb`
  - `final/main.tex`
  - `final/README.md`
  - text extract từ `final/report.pdf`

## 16. Nếu bắt đầu chat mới thì nên làm gì trước

Nếu assistant mới đọc file này, nên làm theo thứ tự:

1. Mở hoặc đọc `summary.md`.
2. Kiểm tra nhanh các file chính:
   - `final/notebook.ipynb`
   - `final/main.tex`
   - `final/report.pdf`
   - `final/predictions.csv`
   - `final/README.md`
3. Nếu người dùng muốn sửa report:
   - sửa `final/main.tex`.
   - compile lại `main.tex` thành `main.pdf`.
   - copy `main.pdf` thành `report.pdf`.
4. Nếu người dùng muốn sửa notebook:
   - tốt nhất sửa `final/build_final_artifacts.py` rồi chạy lại để regenerate notebook.
   - hoặc sửa trực tiếp `final/notebook.ipynb` nếu chỉ chỉnh text/cell nhỏ.
5. Nếu người dùng muốn chạy trên Colab:
   - dùng `final/notebook.ipynb`.
   - upload cùng folder:
     - `ml_dataset_v2_train.csv`
     - `ml_dataset_v2_dev.csv`
     - `hidden_test_data.csv`
6. Nếu có hidden test final thật:
   - thay `final/hidden_test_data.csv` bằng file hidden final mới.
   - nếu file mới có đủ `part_time_job_hours`, notebook vẫn chạy.
   - nếu file mới không có label, notebook vẫn tạo predictions nhưng không tính local hidden metric.
7. Nếu sửa tên nhóm/thành viên:
   - sửa trong `final/main.tex`.
   - có thể sửa thêm title markdown trong notebook nếu muốn.

## 17. Lưu ý quan trọng cho assistant mới

- Trả lời người dùng bằng tiếng Việt nếu không có yêu cầu khác.
- Artifact final như report và notebook vẫn viết tiếng Anh.
- Đừng tự đổi notebook về path phức tạp; người dùng đã yêu cầu đơn giản như giữa kì.
- Đừng thêm chữ `from scratch`; người dùng đã yêu cầu bỏ.
- Đừng dùng sklearn trong notebook.
- Đừng overwrite `report.pdf` bằng ReportLab; report chính là LaTeX.
- Nếu chạy `build_final_artifacts.py`, sau đó vẫn cần compile `main.tex` nếu muốn cập nhật PDF report.
- Nếu cần chỉnh model/report metrics, dùng `final/results/model_comparison.csv` và `summary.json` làm nguồn tham khảo.
- Nếu cần nộp chính thức, kiểm tra lại tên nhóm/thành viên vì hiện vẫn là `Student Team`.

