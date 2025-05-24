import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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
results = []
n_trees_list = [50, 100, 200, 500]

for n in n_trees_list:
    print(f"\n=== Random Forest (n_estimators={n}) ===")
    model = RandomForestClassifier(
        n_estimators=n,
        criterion='entropy',
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)

    # --- Tahminler ---
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    y_test_bin = label_binarize(y_test, classes=class_names)

    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)
    y_train_bin = label_binarize(y_train, classes=class_names)

    # --- Metrikler ---
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    auc = roc_auc_score(y_test_bin, y_prob, average='macro', multi_class='ovr')

    # --- Specificity Hesaplama ---
    specificities = []
    for label in class_names:
        binary_y_test = (y_test == label).astype(int)
        binary_y_pred = (y_pred == label).astype(int)
        tn, fp, fn, tp = confusion_matrix(binary_y_test, binary_y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        specificities.append(specificity)
    specificity_macro = np.mean(specificities)

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.title(f"Confusion Matrix (RFC{n})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # --- ROC Eğrileri ---
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    for i, class_label in enumerate(class_names):
        fpr_test, tpr_test, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        fpr_train, tpr_train, _ = roc_curve(y_train_bin[:, i], y_train_prob[:, i])
        axs[0].plot(fpr_train, tpr_train, label=f'{class_label} (Train AUC={roc_auc_score(y_train_bin[:, i], y_train_prob[:, i]):.2f})')
        axs[1].plot(fpr_test, tpr_test, label=f'{class_label} (Test AUC={roc_auc_score(y_test_bin[:, i], y_prob[:, i]):.2f})')

    axs[0].plot([0, 1], [0, 1], 'k--'); axs[1].plot([0, 1], [0, 1], 'k--')
    axs[0].set_title(f"Train ROC Curve (RFC{n})")
    axs[1].set_title(f"Test ROC Curve (RFC{n})")
    for ax in axs:
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
    plt.tight_layout()
    plt.show()

    # --- Sonuçları Kaydet ---
    results.append({
        'n_estimators': n,
        'Accuracy': acc,
        'Precision': prec,
        'Recall (Sensitivity)': recall,
        'F1-Score': f1,
        'AUC': auc,
        'Specificity': specificity_macro
    })

# ==============================================
# 📋 SONUÇ ÖZETİ – TABLO
# ==============================================

print("\n=== RFC Sonuç Özeti ===")
results_df = pd.DataFrame(results)
print(results_df)

# ==============================================
# 📈 SONUÇ ÖZETİ – GÖRSELLEŞTİRME
# ==============================================

metrics = ['Accuracy', 'Precision', 'Recall (Sensitivity)', 'F1-Score', 'AUC', 'Specificity']
plt.figure(figsize=(10, 6))
for metric in metrics:
    plt.plot(results_df['n_estimators'], results_df[metric], marker='o', label=metric)

plt.title("Random Forest Performans Metrikleri (Test Seti)")
plt.xlabel("n_estimators")
plt.ylabel("Değer")
plt.xticks(results_df['n_estimators'])
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
