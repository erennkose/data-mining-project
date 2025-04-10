import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Veri setini yükle
df = pd.read_excel("dataset for mendeley 181220.xlsx")

print(df.columns)

# Sütun adlarını temizle (boşlukları ve tırnakları sil)
df.columns = df.columns.str.strip().str.replace("'", "", regex=False).str.replace('"', "", regex=False)

# Gerekli sütunları seç
required_columns = [
    'Gender',
    'Age as of Academic Year 17/18',
    'Previous Curriculum (17/18)2',
    'Math20-1 ', 'Science20-1 ', 'English20-1 ',
    'Math20-2 ', 'Science20-2 ', 'English20-2 ',
    'Math20-3 ', 'Science20-3 ', 'English20-3 '
]

df = df[required_columns].dropna()

# Ortalamayı hesapla (tüm 20-1, 20-2, 20-3 sınavları)
exam_cols = [
    'Math20-1 ', 'Science20-1 ', 'English20-1 ',
    'Math20-2 ', 'Science20-2 ', 'English20-2 ',
    'Math20-3 ', 'Science20-3 ', 'English20-3 '
]
df['ExamAverage'] = df[exam_cols].mean(axis=1)

# Performans kategorisini belirle
def get_performance_category(avg):
    if avg >= 85:
        return 'Great'
    elif avg >= 70:
        return 'Good'
    elif avg >= 55:
        return 'Decent'
    elif avg >= 40:
        return 'Bad'
    else:
        return 'Terrible'

df['Performance'] = df['ExamAverage'].apply(get_performance_category)

# X ve y ayır
X = df.drop(['ExamAverage', 'Performance'], axis=1)
y = df['Performance']

# Kategorik sütunları sayısallaştır
X = pd.get_dummies(X, columns=['Gender', 'Previous Curriculum (17/18)2'], drop_first=True)

# Eğitim ve test verisi ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modeli eğit
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Tahmin ve metrikler
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Confusion matrix görselleştirme
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.title("Confusion Matrix")
plt.show()

# Feature importance grafiği
importances = model.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]

class_names = model.classes_
y_test_bin = label_binarize(y_test, classes=class_names)
y_proba = model.predict_proba(X_test)

# Her sınıf için AUC değeri
auc_scores = {}
for i, class_label in enumerate(class_names):
    auc = roc_auc_score(y_test_bin[:, i], y_proba[:, i])
    auc_scores[class_label] = auc

# Sonuçları yazdır
print("\nAUC Skorları (One-vs-Rest):")
for label, score in auc_scores.items():
    print(f"{label}: {score:.4f}")

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=features[indices], palette="viridis")
plt.title("Özellik Önem Dereceleri")
plt.xlabel("Önem")
plt.ylabel("Özellik")
plt.tight_layout()
plt.show()
