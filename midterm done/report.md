# Machine Learning Project Report

**Team Name:** [Your Team Names]

## 1 Introduction

The objective of this project is to develop a machine learning pipeline to predict scholarship eligibility for students. This is a binary classification problem where we use a student’s academic and socio-economic attributes to determine if they receive a scholarship. The goal is to build a model that is both accurate and reproducible, ensuring a fair selection process based on the provided data.

## 2 Dataset

**Place Table 1 here.**

- **What to include:** A simple 2-column table listing the **Feature Name** and a **Brief Description** (e.g., "GPA: Grade point average of the student").

The dataset represents synthetic student records in a hypothetical country. The training set contains **200 samples** and the development (validation) set contains **80 samples**, pre-split by the course. It includes the following features:

- **GPA:** Grade Point Average, on a 2.0–4.0 scale.
- **Attendance Rate:** Proportion of attended classes (0–1).
- **Study Hours Per Week:** Average weekly self-study hours.
- **Exam Score:** Final exam score (0–100).
- **Household Income:** Approximate household monthly income (synthetic units).
- **Target Variable:** A binary label (`label`) where 1 indicates "Scholarship Received" and 0 indicates "Not Received."

## 3 Exploratory Data Analysis

**Place Figure 2 here: Boxplot of GPA**

- **What it shows:** The boxplot comparing GPA for Label 0 vs Label 1.
- **Caption:** "Figure 2: Distribution of GPA scores for scholarship recipients vs non-recipients."

Our EDA focused on understanding feature distributions and relationships:

- **Class Distribution:** The training set contains 122 non-scholarship students (label=0) and 78 scholarship students (label=1), a moderate class imbalance (~61%/39%).
- **Descriptive Statistics:** We used `df.describe()` to find that the data is well-distributed. Household Income showed the highest absolute variance (std=552), while GPA had a smaller range (2.0–4.0).
- **Visualizations:** Using correlation heatmaps, we identified that **GPA** (correlation = 0.68) and **Study Hours Per Week** (correlation = 0.36) are the most influential predictors of scholarship status, while Household Income showed almost no correlation (−0.02).
- **Data Quality:** We identified no missing values, duplicate rows, or extreme outliers that required removal.
- **Insights:** The boxplot of GPA by label revealed a clear separation between classes, suggesting that the criteria for scholarships are primarily merit-based in this synthetic dataset.

## 4 Data Preprocessing

The training (200 samples) and development (80 samples) sets were pre-split by the course. We followed a rigorous preprocessing workflow:

1. **Missing Value Check:** No missing values were detected, ensuring a complete dataset for training.
2. **Categorical Encoding:** As all features were numerical, no encoding was necessary.
3. **Feature ID Removal:** The `id` column was dropped from features to prevent the model from learning from arbitrary identifiers.
4. **Feature Scaling (Standardization):** We used `StandardScaler` to transform all features to have a mean of 0 and a standard deviation of 1. The scaler was fitted on the training data only and applied to dev/test sets to prevent data leakage.
    - **Justification:** Scaling is critical because features like `household_income` (values in the thousands) would otherwise overshadow `gpa` (values 2.0–4.0) in distance-based models like SVM and linear models like Logistic Regression.

## 5 Models

We evaluated four distinct models to find the best fit:

1. **Logistic Regression:** Chosen as a baseline linear model due to its interpretability and efficiency. Parameters: `solver='lbfgs'`, `max_iter=100` (defaults).
2. **SVM (Support Vector Machine):** Chosen for its ability to find the optimal separating hyperplane. Parameters: `kernel='rbf'`, `C=1.0` (defaults), `probability=True`.
3. **Decision Tree:** Chosen to capture non-linear, rule-based relationships. Parameters: `criterion='gini'`, no max depth limit (defaults).
4. **Random Forest:** Chosen to reduce the variance of single decision trees through bagging. Parameters: `n_estimators=100`, `criterion='gini'` (defaults).

**Training Settings:** All models used `random_state=42` to ensure **reproducibility**, as required by the Grading Policy.

## 6 Experimental Results

**Place Table 3 here: Model Performance Comparison**

- **What to include:** This is the most important table. It must have 5 columns:
    1. **Model Name** (Logistic Regression, Decision Tree, etc.)
    2. **Accuracy**
    3. **Precision**
    4. **Recall**
    5. **F1-Score**
- *Note: Use the "Weighted Avg" or "Class 1" numbers from your code output.*

The models were evaluated on the dev set (80 samples).

| Model | Accuracy | Precision (1) | Recall (1) | F1-Score |
| --- | --- | --- | --- | --- |
| **Logistic Regression** | **0.9750** | 0.94 | **1.00** | 0.97 |
| **SVM** | **0.9750** | 0.97 | 0.97 | 0.97 |
| Random Forest | 0.9500 | 0.97 | 0.91 | 0.94 |
| Decision Tree | 0.9000 | 0.85 | 0.91 | 0.88 |

**Justification of Final Choice:** Both Logistic Regression and SVM achieved the highest accuracy (97.5%). We selected **Logistic Regression** for our final submission because it achieved a **perfect Recall (1.00)** for the scholarship class, ensuring that no eligible students were missed by the model.

## 7 Error Analysis

**Place Figure 3 here: Confusion Matrix**

- **What it shows:** The 2x2 grid showing True Positives, False Positives, etc. (The red heatmap we coded).
- **Caption:** "Figure 3: Confusion matrix for the Logistic Regression model, showing two false positive errors."

Our best model (Logistic Regression) made only two errors, both **False Positives** — predicting scholarship for students who did not actually receive one.

- **Analysis:** The two misclassified students (IDs 18 and 67) had moderate-to-high GPAs (2.91 and 3.58) but were non-scholarship students. The model's reliance on GPA as the strongest feature caused it to over-predict scholarship eligibility for these borderline cases.
- **Possible Causes:** The errors likely stem from the linear decision boundary struggling with edge cases near the classification threshold. Because the dataset is synthetic and relatively small (200 training samples), these two errors represent the limit of linear separation for these specific borderline cases.

## 8 Conclusion

The project successfully implemented a complete machine learning workflow. We demonstrated that linear models (Logistic Regression and SVM) are highly effective for this dataset, outperforming tree-based methods.

- **Main Finding:** Academic merit (GPA, with a correlation of 0.68) and study effort (Study Hours Per Week, 0.36) are the primary drivers of scholarship status, while Household Income has almost no predictive power (−0.02).
- **Future Improvements:** Implementing feature engineering such as polynomial features or interaction terms between GPA and study hours could help the model better handle borderline cases, potentially eliminating the final 2.5% of error.