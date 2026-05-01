# Kế hoạch triển khai Final Machine Learning Project

> Dành cho người làm bài hoặc agent tiếp tục làm bài: bám theo từng mục trong file này, đánh dấu xong từng phần, không bỏ qua phần kiểm tra cuối cùng.

**Mục tiêu:** Hoàn thành bài cuối kì Machine Learning về hệ thống dự đoán học bổng sinh viên, gồm report, notebook, predictions và các file hỗ trợ.

**Cách tiếp cận:** Dựng lại baseline từ bài midterm đã làm, mở rộng sang dataset final v2, tự code thuật toán học máy bằng `numpy` thay vì dùng thư viện train ML như `scikit-learn`, so sánh nhiều model, thêm cải tiến rõ ràng, rồi viết report tiếng Anh tối thiểu 25 trang.

**Tech stack được dùng:** Python, pandas, numpy, matplotlib, seaborn, reportlab, nbformat. Không dùng `scikit-learn` để train model.

---

## 1. Hiểu đúng yêu cầu đề cuối kì

File đề chính:

- `final_handout.pdf`
- `ml_project_student_guide.pdf`

Deadline ghi trong đề:

- 15/05/2026

Team policy:

- Làm theo nhóm 2 người.
- Report cần có phần contribution của từng thành viên.

Bài toán:

- Dự đoán một sinh viên có nhận học bổng hay không.
- Đây là bài toán binary classification.
- Target là `label`.
- `label = 1` nghĩa là có học bổng.
- `label = 0` nghĩa là không có học bổng.

Yêu cầu bắt buộc trong final:

- Baseline system dựa trên midterm.
- Exploratory Data Analysis.
- Data preprocessing.
- Ít nhất 3 machine learning models.
- Có ít nhất 1 linear model, nên dùng Logistic Regression.
- Có ít nhất 1 tree-based model, ví dụ Decision Tree hoặc Random Forest.
- Có ít nhất 1 model cổ điển khác, ví dụ KNN hoặc SVM.
- Có ít nhất 1 cải tiến có ý nghĩa so với midterm.
- Đánh giá bằng Accuracy, Precision, Recall, F1-score, Confusion Matrix.
- Nên có ROC-AUC hoặc PR-AUC nếu phù hợp.
- Error analysis.
- Model interpretation.
- Final recommendation.

File phải nộp:

- `report.pdf`
- `notebook.ipynb`
- `predictions.csv`

File khuyến nghị nên có:

- `README.md`
- `requirements.txt`

Yêu cầu report:

- Viết bằng tiếng Anh.
- Có figure, table và thảo luận kỹ thuật.
- Trang bìa ghi tối thiểu 25 trang.
- Một đoạn khác ghi tối thiểu 15 trang.
- Để an toàn, làm theo mức cao hơn: tối thiểu 25 trang.
- Không được chỉ paste code hoặc screenshot.
- Phải giải thích "đã làm gì" và "vì sao làm như vậy".

---

## 2. Trạng thái folder hiện tại

Folder gốc hiện có:

- `final/`
- `midterm/`
- `midterm done/`
- `final_handout.pdf`
- `ml_project_student_guide.pdf`

Folder `final/` hiện có:

- `final/data/ml_dataset_v2_train.csv`
- `final/data/ml_dataset_v2_dev.csv`
- `final/notebook/ml_final_project_notebook_template.ipynb`

Folder `midterm done/` hiện có:

- `notebook.ipynb`
- `report.pdf`
- `report.docx`
- `report.md`
- `predictions.csv`
- `hidden_test_data.csv`
- dataset v1 train/dev
- tài liệu midterm

Trạng thái hiện tại sau khi hoàn thiện:

- `final/report.pdf` đã được compile từ `final/main.tex`.
- `final/notebook.ipynb` đã được tạo thành notebook hoàn chỉnh, chia cell nhỏ.
- `final/predictions.csv` đã đúng format `id,label_pred`.
- `final/README.md` và `final/requirements.txt` vẫn được giữ làm file hỗ trợ.
- `submission/` là folder nộp chính, chỉ chứa 3 file bắt buộc:
  - `submission/report.pdf`
  - `submission/notebook.ipynb`
  - `submission/predictions.csv`

Dataset final dùng khi chạy notebook:

- `final/ml_dataset_v2_train.csv`
- `final/ml_dataset_v2_dev.csv`
- `final/hidden_test_data.csv`

Các bản trong `final/data/` vẫn được giữ lại như dữ liệu gốc tham chiếu.

---

## 3. Phân tích dataset final

Final train:

- File: `final/data/ml_dataset_v2_train.csv`
- Số dòng: 250
- Số cột: 8
- Không có missing values.
- Phân bố label:
  - label 0: 150 dòng
  - label 1: 100 dòng
  - tỉ lệ positive: 40%

Final dev:

- File: `final/data/ml_dataset_v2_dev.csv`
- Số dòng: 100
- Số cột: 8
- Không có missing values.
- Phân bố label:
  - label 0: 60 dòng
  - label 1: 40 dòng
  - tỉ lệ positive: 40%

Các cột final:

- `id`
- `gpa`
- `attendance_rate`
- `study_hours_per_week`
- `exam_score`
- `household_income`
- `part_time_job_hours`
- `label`

Điểm mới so với midterm:

- Final có thêm cột `part_time_job_hours`.
- Train final tăng từ 200 lên 250 dòng.
- Dev final tăng từ 80 lên 100 dòng.
- `household_income` trong final có range rộng hơn.
- `study_hours_per_week` trong final có max cao hơn.

Lưu ý schema hidden test:

- File `final/hidden_test_data.csv` có thể không có `part_time_job_hours`.
- Khi tạo `predictions.csv`, workflow tự thêm cột thiếu bằng median của final train.
- Trong report và notebook nộp, gọi thống nhất là hidden test, không dùng cách gọi cũ gây hiểu nhầm.

---

## 4. Ràng buộc kỹ thuật

Không được dùng:

- `scikit-learn` để train model.
- `xgboost`, `lightgbm`, hoặc thư viện train ML có sẵn.

Được dùng:

- `pandas` để đọc và xử lý bảng.
- `numpy` để tính toán thuật toán.
- `matplotlib` và `seaborn` để vẽ hình.
- `reportlab` để tạo PDF report.
- `nbformat` để tạo notebook.

Các thuật toán sẽ tự code:

- Standard scaler.
- Logistic Regression.
- Decision Tree.
- Random Forest.
- KNN.
- Metrics: accuracy, precision, recall, F1, confusion matrix.
- Permutation importance.
- Feature importance cho Random Forest.
- Threshold tuning cho Logistic Regression.

---

## 5. Chiến lược model để đạt điểm tốt và dễ giải thích

Model 1: Logistic Regression

- Vai trò: linear model bắt buộc.
- Lợi thế: dễ giải thích bằng coefficient.
- Dùng làm baseline chính.
- Có thể tune learning rate, số vòng lặp, threshold.

Model 2: Decision Tree

- Vai trò: tree-based model.
- Lợi thế: dễ giải thích bằng luật chia nhánh.
- Có thể tune max depth, min samples split.

Model 3: Random Forest

- Vai trò: tree-based ensemble.
- Lợi thế: thường ổn định hơn Decision Tree.
- Có thể dùng feature importance.
- Dùng làm ứng viên mạnh.

Model 4: KNN

- Vai trò: classical model khác.
- Lợi thế: dễ hiểu, dễ tự code.
- Cần scaling.
- Có thể tune số neighbor `k`.

Lý do không chọn SVM làm model chính:

- SVM tự code chuẩn có nhiều chi tiết tối ưu hơn.
- KNN dễ giải thích hơn cho report và phù hợp với ràng buộc không dùng scikit-learn.
- Nếu cần thêm, có thể nói midterm từng dùng SVM, còn final dùng KNN để bổ sung model theo hướng distance-based.

---

## 6. Cải tiến final so với midterm

Cải tiến 1: Dataset v2 và feature mới

- Final dùng thêm `part_time_job_hours`.
- Report cần phân tích xem feature này ảnh hưởng như thế nào.

Cải tiến 2: Hyperparameter tuning có kiểm soát

- Chia một phần train thành validation nội bộ.
- Tune các model trên validation nội bộ.
- Sau đó train lại trên full train và đánh giá trên dev.
- Không tune trực tiếp quá mức trên dev để tránh leakage.

Cải tiến 3: Feature engineering

Tạo thêm các feature dễ hiểu:

- `academic_strength = gpa * exam_score`
- `study_efficiency = exam_score / (study_hours_per_week + 1)`
- `income_per_study_hour = household_income / (study_hours_per_week + 1)`
- `work_study_balance = study_hours_per_week - part_time_job_hours`

Lưu ý:

- Feature engineering phải dùng dữ liệu feature, không dùng label.
- Không được tạo feature bằng cách nhìn vào đáp án.

Cải tiến 4: Threshold tuning cho Logistic Regression

- Threshold mặc định là 0.5.
- Có thể thử 0.35, 0.40, 0.45, 0.50, 0.55, 0.60.
- Chọn threshold theo F1 hoặc ưu tiên recall nếu muốn tránh bỏ sót sinh viên xứng đáng.

Cải tiến 5: Error analysis chi tiết hơn

- Tách false positive và false negative.
- So sánh mean feature values của nhóm đúng và nhóm sai.
- Trích một vài case sai tiêu biểu.

Cải tiến 6: Model interpretation

- Logistic Regression coefficients.
- Random Forest feature importance.
- Permutation importance.

---

## 7. Cấu trúc file cần tạo

Tạo hoặc cập nhật:

- `final/notebook.ipynb`
  - Notebook hoàn chỉnh, chạy từ đầu đến cuối.
  - Code và markdown bằng tiếng Anh.
  - Không dùng scikit-learn để train.

- `final/predictions.csv`
  - Format đúng: `id,label_pred`.
  - Dùng `final/hidden_test_data.csv`.
  - Nếu hidden thiếu `part_time_job_hours`, tự thêm bằng median từ train final.

- `final/report.pdf`
  - Report tiếng Anh.
  - Tối thiểu 25 trang.
  - Có tables, figures, metrics, error analysis, interpretation.

- `final/README.md`
  - Hướng dẫn chạy notebook.
  - Ghi rõ cách đặt train/dev/hidden cùng folder notebook.
  - Ghi rõ không dùng scikit-learn để train.

- `final/requirements.txt`
  - Ghi các thư viện cần cài.

- `final/figures/`
  - Chứa hình dùng trong report.

- `final/results/`
  - Chứa bảng metrics và output hỗ trợ.

---

## 8. Kế hoạch làm chi tiết

### Task 1: Xác nhận dữ liệu và yêu cầu

- [ ] Đọc `final_handout.pdf`.
- [ ] Đọc `ml_project_student_guide.pdf`.
- [ ] Đọc `midterm done/report.md`.
- [ ] Đọc `midterm done/notebook.ipynb`.
- [ ] Đọc `final/data/ml_dataset_v2_train.csv`.
- [ ] Đọc `final/data/ml_dataset_v2_dev.csv`.
- [ ] Đọc `final/hidden_test_data.csv`.
- [ ] Ghi lại file phải nộp.
- [ ] Ghi lại file final đang thiếu.

Kết quả mong muốn:

- Biết chính xác bài yêu cầu gì.
- Biết final khác midterm ở đâu.
- Biết cần tạo những artifact nào.

### Task 2: Xây notebook final

- [ ] Tạo `final/notebook.ipynb`.
- [ ] Viết phần setup.
- [ ] Load train/dev/hidden.
- [ ] Validate columns.
- [ ] Thêm compatibility handling cho hidden test nếu thiếu feature final.
- [ ] EDA dataset size.
- [ ] EDA descriptive statistics.
- [ ] EDA target distribution.
- [ ] EDA correlation heatmap.
- [ ] EDA boxplot theo label.
- [ ] EDA data quality checks.
- [ ] Tạo feature engineering.
- [ ] Scale features bằng scaler tự code.
- [ ] Code Logistic Regression từ đầu.
- [ ] Code Decision Tree từ đầu.
- [ ] Code Random Forest từ đầu.
- [ ] Code KNN từ đầu.
- [ ] Code metric functions từ đầu.
- [ ] Tune hyperparameters.
- [ ] Train lại model tốt trên full train.
- [ ] Evaluate trên dev.
- [ ] Vẽ confusion matrix.
- [ ] Làm error analysis.
- [ ] Làm model interpretation.
- [ ] Tạo `predictions.csv`.

Kết quả mong muốn:

- Notebook chạy được từ đầu đến cuối.
- Có đủ nội dung để report không bị thiếu.
- Có đủ output để kiểm tra.

### Task 3: Tạo predictions

- [ ] Dùng best model từ final.
- [ ] Load `final/hidden_test_data.csv`.
- [ ] Nếu hidden có label thì không dùng label làm feature.
- [ ] Nếu hidden thiếu `part_time_job_hours`, thêm bằng median của train final.
- [ ] Predict label.
- [ ] Xuất `final/predictions.csv`.
- [ ] Kiểm tra đúng 2 cột: `id,label_pred`.
- [ ] Kiểm tra label_pred chỉ có 0 hoặc 1.

Kết quả mong muốn:

- File nộp đúng format.
- Có thể thay hidden test final mới vào sau này.

### Task 4: Tạo report tiếng Anh

- [ ] Tạo `final/report.pdf`.
- [ ] Report tối thiểu 25 trang.
- [ ] Có title page.
- [ ] Có abstract hoặc executive summary.
- [ ] Có table of contents đơn giản.
- [ ] Có introduction.
- [ ] Có dataset description.
- [ ] Có EDA.
- [ ] Có preprocessing.
- [ ] Có model descriptions.
- [ ] Có baseline section.
- [ ] Có improvement strategy.
- [ ] Có experimental results.
- [ ] Có error analysis.
- [ ] Có model interpretation.
- [ ] Có final recommendation.
- [ ] Có limitations.
- [ ] Có reproducibility notes.
- [ ] Có contribution statement.
- [ ] Có appendix.

Kết quả mong muốn:

- Report đủ dài, rõ, có tính học thuật.
- Không chỉ là copy output từ notebook.
- Có giải thích vì sao làm từng quyết định.

### Task 5: Tạo README và requirements

- [ ] Tạo `final/README.md`.
- [ ] Ghi cách chạy notebook.
- [ ] Ghi dataset nào dùng để train/dev.
- [ ] Ghi cách đặt `hidden_test_data.csv` cùng folder notebook.
- [ ] Ghi file cần nộp.
- [ ] Ghi constraint không dùng scikit-learn.
- [ ] Tạo `final/requirements.txt`.

Kết quả mong muốn:

- Người khác đọc vào biết chạy bài.
- Giáo viên có thể tái lập kết quả.

### Task 6: Kiểm tra cuối

- [ ] Kiểm tra `final/report.pdf` tồn tại.
- [ ] Kiểm tra report có ít nhất 25 trang.
- [ ] Kiểm tra `final/notebook.ipynb` tồn tại.
- [ ] Kiểm tra `final/predictions.csv` tồn tại.
- [ ] Kiểm tra `final/README.md` tồn tại.
- [ ] Kiểm tra `final/requirements.txt` tồn tại.
- [ ] Chạy script hoặc notebook workflow để đảm bảo không lỗi.
- [ ] Đọc `predictions.csv` và kiểm tra format.
- [ ] Kiểm tra không import scikit-learn trong notebook.

---

## 9. Cấu trúc report đề xuất

1. Title Page
2. Abstract
3. Team Contribution Statement
4. Introduction
5. Problem Statement
6. Dataset Description
7. Comparison with Midterm Baseline
8. Exploratory Data Analysis
9. Data Quality Assessment
10. Preprocessing Methodology
11. Feature Engineering
12. Baseline System
13. Manual NumPy Model Implementations
14. Hyperparameter Tuning Strategy
15. Experimental Setup
16. Model Evaluation Metrics
17. Experimental Results
18. Confusion Matrix Analysis
19. Error Analysis
20. Model Interpretation
21. Practical Considerations
22. Final Model Recommendation
23. Limitations
24. Reproducibility
25. Conclusion
26. Appendix

---

## 10. Tiêu chí tự chấm trước khi nộp

Problem understanding:

- [ ] Nói rõ task là binary classification.
- [ ] Nói rõ target là `label`.
- [ ] Nói rõ dữ liệu là synthetic student scholarship data.

EDA:

- [ ] Có dataset size.
- [ ] Có feature types.
- [ ] Có descriptive statistics.
- [ ] Có target distribution.
- [ ] Có visualizations.
- [ ] Có data quality discussion.

Preprocessing:

- [ ] Drop `id` khỏi feature.
- [ ] Không dùng label để tạo feature.
- [ ] Fit scaler trên train only.
- [ ] Transform dev/hidden bằng scaler train.
- [ ] Giải thích tránh data leakage.

Model development:

- [ ] Có Logistic Regression.
- [ ] Có Decision Tree hoặc Random Forest.
- [ ] Có KNN hoặc model classical khác.
- [ ] Tất cả model train bằng thuật toán tự code.

Improvement:

- [ ] Nêu rõ final cải tiến gì so với midterm.
- [ ] Có evidence bằng metric hoặc analysis.

Evaluation:

- [ ] Có Accuracy.
- [ ] Có Precision.
- [ ] Có Recall.
- [ ] Có F1.
- [ ] Có Confusion Matrix.
- [ ] Có table so sánh model.

Error analysis:

- [ ] Có false positives.
- [ ] Có false negatives.
- [ ] Có ví dụ hoặc pattern sai.
- [ ] Có giới hạn hệ thống.

Interpretation:

- [ ] Có coefficients hoặc feature importance.
- [ ] Có giải thích feature nào quan trọng.
- [ ] Không nói quan hệ nhân quả khi chỉ có correlation.

Submission:

- [ ] `report.pdf`
- [ ] `notebook.ipynb`
- [ ] `predictions.csv`
- [ ] `README.md`
- [ ] `requirements.txt`

---

## 11. Rủi ro và cách xử lý

Rủi ro 1: Hidden test khác schema train final.

- Cách xử lý hiện tại: notebook đọc `final/hidden_test_data.csv`.
- Nếu hidden test thiếu `part_time_job_hours`, workflow thêm cột bằng median từ train final.
- Nếu có hidden test mới, chỉ cần thay file `final/hidden_test_data.csv` rồi chạy lại notebook.

Rủi ro 2: Thông tin nhóm/thành viên cần được giữ đúng khi sửa report.

- Report hiện đã có:
  - Chau Pham Tuan Kiet - 523K0011
  - Chau Thanh Vu - 523K0032
  - Tran Anh Minh - 523K0017
- Đề có Team Policy yêu cầu contribution statement; report đã có mục này và đã chia vai trò theo từng thành viên.

Rủi ro 3: Không dùng scikit-learn nên model tự code có thể đơn giản hơn.

- Cách xử lý: chọn model dễ tự code, dễ giải thích, đủ theo yêu cầu.
- Ưu tiên Logistic Regression, Decision Tree, Random Forest, KNN.

Rủi ro 4: Report thiếu trang.

- Cách xử lý: tạo report tối thiểu 25 trang, thêm appendix và discussion đủ rõ.

Rủi ro 5: Overfitting vì dataset nhỏ.

- Cách xử lý: dùng dev set cẩn thận, không claim quá mức, ghi limitations.

---

## 12. Checklist cuối cùng trước khi gửi giáo viên

- [ ] Mở `final/report.pdf` và kiểm tra số trang.
- [ ] Mở `final/notebook.ipynb` và chạy lại toàn bộ.
- [ ] Mở `final/predictions.csv` và kiểm tra header đúng `id,label_pred`.
- [ ] Đảm bảo report viết bằng tiếng Anh.
- [ ] Đảm bảo notebook code/markdown bằng tiếng Anh.
- [ ] Đảm bảo không có dòng import `sklearn`.
- [ ] Đảm bảo README nói rõ cách tái lập.
- [ ] Đảm bảo contribution statement và tên nhóm/thành viên vẫn đúng sau khi compile lại PDF.
- [ ] Nộp folder `submission/` hoặc 3 file bắt buộc: `report.pdf`, `notebook.ipynb`, `predictions.csv`.

---

## 13. Cập nhật audit cuối cùng trước khi nộp

Trạng thái đã kiểm tra:

- `submission/` có đúng 3 file bắt buộc:
  - `report.pdf`
  - `notebook.ipynb`
  - `predictions.csv`
- `report.pdf` được compile lại từ `final/main.tex`.
- `report.pdf` có 31 trang.
- Text trong `report.pdf` không còn các cụm mô tả hidden test theo cách cũ.
- `notebook.ipynb` có 35 cells, trong đó 17 code cells.
- Notebook chạy từ đầu đến cuối thành công trong folder `final/`.
- Notebook không import `sklearn`.
- Notebook không có absolute Windows path.
- `predictions.csv` có đúng 2 cột: `id,label_pred`.
- `predictions.csv` có 100 dòng, không null, không trùng id, label chỉ gồm 0 và 1.

Đối chiếu rubric 10 điểm:

- Problem understanding and dataset description: đã có Introduction, Dataset, target, binary classification.
- EDA: đã có size, descriptive statistics, target distribution, correlation, figure theo feature.
- Data preprocessing: đã có drop `id`, scaler fit train only, feature engineering, hidden schema compatibility.
- Model development: đã có Logistic Regression, Decision Tree, Random Forest, KNN.
- System improvement: đã có dataset v2, feature engineering, tuning, threshold selection, interpretation.
- Evaluation and comparison: đã có Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix, bảng so sánh.
- Error analysis and interpretation: đã có error cases, error pattern, coefficients, random forest importance, permutation importance.
- Report quality and reproducibility: report LaTeX rõ ràng, notebook chạy được, README/requirements hỗ trợ.

Thông tin nhóm đã được điền:

- `final/main.tex` và `submission/report.pdf` đã có đủ 3 thành viên và MSSV.
- Contribution statement đã chia đều theo vai trò:
  - Chau Pham Tuan Kiet: file submission, path/data checks, prediction format, notebook cleanup, reproducibility checks.
  - Chau Thanh Vu: preprocessing, feature engineering, scaling, Logistic Regression, KNN, tuning.
  - Tran Anh Minh: EDA, tree-based models, model comparison, error analysis, interpretation, report polish.
