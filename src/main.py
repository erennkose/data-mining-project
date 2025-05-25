import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================
# 🔢 VERİ YÜKLEME VE TEMİZLEME
# ==============================================

df = pd.read_excel("dataset for mendeley 181220.xlsx")
df.columns = df.columns.str.strip().str.replace("'", "", regex=False).str.replace('"', "", regex=False)

required_columns = [
    'Gender',
    'Age as of Academic Year 17/18',
    'Previous Curriculum (17/18)2',
    'Math20-1 ', 'Science20-1 ', 'English20-1 ',
    'Math20-2 ', 'Science20-2 ', 'English20-2 ',
    'Math20-3 ', 'Science20-3 ', 'English20-3 '
]
df = df[required_columns].dropna()

exam_cols = [col for col in df.columns if "Math" in col or "Science" in col or "English" in col]
df['ExamAverage'] = df[exam_cols].mean(axis=1)

def get_placement_level(avg):
    if avg >= 85:
        return 'High'
    elif avg >= 75:
        return 'Medium'
    else:
        return 'Low'

df['PlacementLevel'] = df['ExamAverage'].apply(get_placement_level)

# ==============================================
# 📊 VERİ MADENCİLİĞİ GÖRSELLEŞTİRME (EDA)
# ==============================================

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='PlacementLevel', order=['Low', 'Medium', 'High'], palette='Set2')
plt.title("PlacementLevel Sınıf Dağılımı")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.histplot(df['ExamAverage'], kde=True, bins=20, color='skyblue')
plt.title("ExamAverage Dağılımı")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x='Gender', y='ExamAverage', palette='pastel')
plt.title("Cinsiyete Göre Ortalama Notlar")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
sns.boxplot(data=df, x='Previous Curriculum (17/18)2', y='ExamAverage', palette='muted')
plt.title("Müfredata Göre Ortalama Notlar")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
corr = df.select_dtypes(include='number').corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title("Sayısal Özellikler Arası Korelasyon Matrisi")
plt.tight_layout()
plt.show()

# ==============================================
# 🎯 MODELLEME – RANDOM FOREST
# ==============================================

X = df.drop(['ExamAverage', 'PlacementLevel'], axis=1)
y = df['PlacementLevel']
X = pd.get_dummies(X, columns=['Gender', 'Previous Curriculum (17/18)2'], drop_first=True)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=2/3, random_state=42, stratify=y_temp)

class_names = np.unique(y)

# ===============================
# 🔍 EN İYİ RFC PARAMETRELERİNİ BUL (GRID SEARCH)
# ===============================

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2'],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(bootstrap=True, random_state=42, n_jobs=-1),
    param_grid=param_grid,
    scoring='f1_macro',
    cv=3,
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\nEn iyi parametreler (GridSearchCV ile):")
best_params = grid_search.best_params_
print(best_params)

# ===============================
# 🏆 EN İYİ MODEL İLE EĞİTİM VE TEST
# ===============================

best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)
y_test_bin = label_binarize(y_test, classes=class_names)

y_train_pred = best_model.predict(X_train)
y_train_prob = best_model.predict_proba(X_train)
y_train_bin = label_binarize(y_train, classes=class_names)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
auc = roc_auc_score(y_test_bin, y_prob, average='macro', multi_class='ovr')

specificities = []
for label in class_names:
    binary_y_test = (y_test == label).astype(int)
    binary_y_pred = (y_pred == label).astype(int)
    tn, fp, fn, tp = confusion_matrix(binary_y_test, binary_y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    specificities.append(specificity)
specificity_macro = np.mean(specificities)

print("\nEn iyi model test sonuçları:")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall (Sensitivity): {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"AUC: {auc:.4f}")
print(f"Specificity: {specificity_macro:.4f}")

cm = confusion_matrix(y_test, y_pred, labels=class_names)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Greens')
plt.title("En İyi Model - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(14, 5))
for i, class_label in enumerate(class_names):
    fpr_test, tpr_test, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    fpr_train, tpr_train, _ = roc_curve(y_train_bin[:, i], y_train_prob[:, i])
    axs[0].plot(fpr_train, tpr_train, label=f'{class_label} (Train AUC={roc_auc_score(y_train_bin[:, i], y_train_prob[:, i]):.2f})')
    axs[1].plot(fpr_test, tpr_test, label=f'{class_label} (Test AUC={roc_auc_score(y_test_bin[:, i], y_prob[:, i]):.2f})')

axs[0].plot([0, 1], [0, 1], 'k--'); axs[1].plot([0, 1], [0, 1], 'k--')
axs[0].set_title("Train ROC Curve (Best RFC)")
axs[1].set_title("Test ROC Curve (Best RFC)")
for ax in axs:
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
plt.tight_layout()
plt.show()

# ==============================================
# 📈 PERFORMANS METRİKLERİ GÖRSELLEŞTİRME
# ==============================================

metrics = {
    "Accuracy": acc,
    "Precision": prec,
    "Recall (Sensitivity)": recall,
    "F1-Score": f1,
    "AUC": auc,
    "Specificity": specificity_macro
}

plt.figure(figsize=(8, 5))
sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette='viridis')
plt.ylim(0, 1.05)
plt.ylabel("Skor")
plt.title("Model Performans Metrikleri (Test Verisi)")
for i, v in enumerate(metrics.values()):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', va='bottom', fontweight='bold')
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# ==============================================
# ÖZNİTELİK ÖNEMİ GÖRSELLEŞTİRME (FEATURE IMPORTANCE)
# ==============================================

importances = best_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance_df.head(50), x='Importance', y='Feature', palette='mako')
plt.title("Öznitelik Önem Sıralamaları(Random Forest)")
plt.xlabel("Önem Skoru")
plt.ylabel("Öznitelik")
plt.tight_layout()
plt.show()
