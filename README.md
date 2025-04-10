# 🧠 Student Performance Classification using Random Forest

This project is a data mining application developed for the **BLM0463 - Introduction to Data Mining** course.  
The aim is to classify students' academic performance levels using a machine learning model based on their historical exam results.

## 📌 Project Description

The dataset includes multiple academic term exam scores and student-related features. The goal is to predict **student success levels** categorized as:

- `Great`
- `Good`
- `Decent`
- `Bad`
- `Terrible`

The classification model is built using a **Random Forest Classifier**, and its performance is evaluated with standard metrics such as:

- Accuracy  
- Sensitivity & Specificity  
- F1-Score  
- Confusion Matrix  
- AUC Scores (One-vs-Rest)

---

## 📊 Dataset Information

- **Source:** [Dataset of student level prediction in UAE](https://www.sciencedirect.com/science/article/pii/S235234092100192X)
- **Fields used:**
  - Demographics: `Gender`, `Age`, `Previous Curriculum`
  - Academic scores: `Math20-1`, `Science20-2`, `English20-3`, etc. (9 total exams)
- **Target variable:** Average of all exam scores ➝ mapped to 5-class performance level

---

## ⚙️ Technologies & Libraries

- Python 3.12
- Pandas
- NumPy
- Scikit-learn
- Seaborn & Matplotlib

---

## 📈 Model Overview

- **Model:** Random Forest Classifier  
- **Train/Test Split:** 70% / 30%  
- **Encoding:** One-hot encoding for categorical features  
- **Target Mapping:** Exam average → Performance category  

---

## 📌 Evaluation Metrics

- `classification_report` to get:
  - Accuracy
  - Precision
  - Recall (Sensitivity)
  - F1-Score
- `confusion_matrix` plotted using Seaborn
- `roc_auc_score` using one-vs-rest approach for multiclass AUC

---

## 📊 Visualizations

- Confusion Matrix (Heatmap)
- Feature Importance (Barplot)
- AUC Scores per Class (Barplot)

---

## 🧪 How to Run

### 1. Clone the repo:
```bash
git clone https://github.com/your-username/student-performance-classification.git
cd student-performance-classification 
```

### 2. Install required packages:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl
```

### 3. Run the model:
```bash
python src/main.py
```

---
## 💻 Author
- Eren KÖSE - Computer Engineering, 3rd year
