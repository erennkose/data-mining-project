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

# Veri yükleme ve hazırlık (önceki koddan aynen alındı)
df = pd.read_excel("dataset for mendeley 181220.xlsx")
df.columns = df.columns.str.strip().str.replace("'", "", regex=False).str.replace('"', "", regex=False)

# Kullanılan sütunlar
required_columns = [
    'Gender',
    'Age as of Academic Year 17/18',
    'Previous Curriculum (17/18)2',
    'Math20-1 ', 'Science20-1 ', 'English20-1 ',
    'Math20-2 ', 'Science20-2 ', 'English20-2 ',
    'Math20-3 ', 'Science20-3 ', 'English20-3 '
]
df = df[required_columns].dropna()

# Ortalama ve seviye
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

# Özellikler ve hedef
X = df.drop(['ExamAverage', 'PlacementLevel'], axis=1)
y = df['PlacementLevel']

# Kategorik verileri kodla
X = pd.get_dummies(X, columns=['Gender', 'Previous Curriculum (17/18)2'], drop_first=True)

# Train (70%) + temp (30%) ayırımı
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Temp'ten validation (10%) ve test (20%) ayırımı
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=2/3, random_state=42, stratify=y_temp)

# Sınıflar
class_names = np.unique(y)

# Sonuçları tutmak için liste
results = []

# Test edilecek farklı ağaç sayıları
n_trees_list = [50, 100, 200, 500]

for n in n_trees_list:
    print(f"\n=== Random Forest (n_estimators={n}) ===")
    model = RandomForestClassifier(n_estimators=n, random_state=42)
    model.fit(X_train, y_train)

    # Tahminler
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    y_test_bin = label_binarize(y_test, classes=class_names)

    # Metrikler
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    auc = roc_auc_score(y_test_bin, y_prob, average='macro', multi_class='ovr')

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.title(f"Confusion Matrix (RFC{n})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # ROC Eğrisi
    plt.figure(figsize=(8, 5))
    for i, class_label in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        plt.plot(fpr, tpr, label=f'{class_label} (AUC={roc_auc_score(y_test_bin[:, i], y_prob[:, i]):.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"ROC Curve (RFC{n})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Sonuçları kaydet
    results.append({
        'n_estimators': n,
        'Accuracy': acc,
        'Precision': prec,
        'Recall (Sensitivity)': recall,
        'F1-Score': f1,
        'AUC': auc
    })

# Sonuçları tablo halinde yazdır
print("\n=== RFC Sonuç Özeti ===")
results_df = pd.DataFrame(results)
print(results_df)
