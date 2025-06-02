# 🧠 Student Academic Placement Classification using Random Forest

This project is a comprehensive data mining application developed for the **BLM0463 - Introduction to Data Mining** course. The aim is to classify students' academic placement levels using machine learning based on their historical exam performance across multiple subjects and academic terms.

## 📌 Project Description

The dataset contains longitudinal academic performance data from students across three academic terms (20-1, 20-2, 20-3) in core subjects. The goal is to predict **student placement levels** categorized as:

- **High** (≥85 average)
- **Medium** (75-84 average)  
- **Low** (<75 average)

The classification model uses **Random Forest Classifier** with comprehensive hyperparameter tuning via GridSearchCV, and performance is evaluated using multiple metrics including sensitivity, specificity, and AUC scores.

---

## 📊 Dataset Information

- **Source:** "dataset for mendeley 181220.xlsx"
- **Key Features:**
  - **Demographics:** Gender, Age (Academic Year 17/18), Previous Curriculum
  - **Academic Performance:** 9 exam scores across 3 terms:
    - Math20-1, Math20-2, Math20-3
    - Science20-1, Science20-2, Science20-3  
    - English20-1, English20-2, English20-3
- **Target Variable:** Average of all 9 exam scores → mapped to 3-class placement level
- **Data Preprocessing:** Missing values removed, categorical variables one-hot encoded

---

## ⚙️ Technologies & Libraries

- **Python 3.x**
- **Core Libraries:**
  - `pandas` & `numpy` - Data manipulation
  - `scikit-learn` - Machine learning pipeline
  - `matplotlib` & `seaborn` - Data visualization
- **Key ML Components:**
  - Random Forest Classifier
  - GridSearchCV for hyperparameter optimization
  - Cross-validation (3-fold)

---

## 🔧 Machine Learning Pipeline

### Data Preprocessing
- Column name cleaning (whitespace and quote removal)
- Missing value removal
- Feature engineering: ExamAverage calculation
- One-hot encoding for categorical variables

### Model Configuration
- **Algorithm:** Random Forest Classifier
- **Data Split:** 70% Train / 10% Validation / 20% Test
- **Stratified sampling** to maintain class distribution
- **Hyperparameter Grid Search:**
  - `n_estimators`: [50, 100, 200]
  - `max_depth`: [5, 10, 15]
  - `min_samples_split`: [2, 5]
  - `min_samples_leaf`: [1, 2]
  - `max_features`: ['sqrt', 'log2']
  - `criterion`: ['gini', 'entropy']

---

## 📈 Evaluation Metrics

### Classification Metrics
- **Accuracy** - Overall classification correctness
- **Precision** (Macro-averaged) - Positive prediction accuracy
- **Recall/Sensitivity** (Macro-averaged) - True positive detection rate
- **Specificity** - True negative detection rate (calculated per class)
- **F1-Score** (Macro-averaged) - Harmonic mean of precision & recall
- **AUC** (One-vs-Rest) - Area under ROC curve for multiclass

### Overfitting Analysis
- ROC curves for both training and test data
- Performance comparison across datasets

---

## 📊 Visualizations & Analysis

### Exploratory Data Analysis (EDA)
- Class distribution visualization
- Exam average distribution (histogram with KDE)
- Gender-based performance comparison (box plots)
- Curriculum-based performance analysis
- Correlation heatmap for numerical features

### Model Performance Visualization
- **Confusion Matrix** - Classification accuracy per class
- **ROC Curves** - Separate plots for train/test data
- **Performance Metrics Bar Chart** - All metrics comparison
- **Feature Importance Plot** - Top 50 most influential features

---

## 🧪 How to Run

### 1. Prerequisites
Ensure you have the dataset file: `dataset for mendeley 181220.xlsx`

Dataset link: https://www.sciencedirect.com/science/article/pii/S235234092100192X

### 2. Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl
```

### 4. Expected Output
- Multiple visualization plots
- Grid search progress (verbose output)
- Best hyperparameters
- Comprehensive performance metrics
- Feature importance rankings

---

## 🎯 Key Results

The model provides:
- **Automated hyperparameter optimization** via GridSearchCV
- **Comprehensive performance evaluation** with 6 key metrics
- **Feature importance analysis** to understand predictive factors
- **Overfitting detection** through train/test ROC comparison
- **Multi-class classification** with balanced evaluation

---

## 📚 Academic Context

This project demonstrates practical application of:
- **Supervised Learning** - Classification with labeled data
- **Model Selection** - Hyperparameter tuning and cross-validation
- **Performance Evaluation** - Multiple metrics for comprehensive assessment
- **Data Visualization** - EDA and results interpretation
- **Feature Engineering** - Creating meaningful predictors from raw data

---

## 💻 Author

**Eren KÖSE** - Computer Engineering, 3rd Year  
*BLM0463 - Introduction to Data Mining Course Project*

---

## 📝 Notes

- The model uses **macro-averaging** for multiclass metrics to give equal weight to all classes
- **Stratified sampling** ensures representative class distribution in train/test splits
- **Feature importance** analysis helps identify the most predictive academic indicators
- All visualizations are automatically displayed during execution
