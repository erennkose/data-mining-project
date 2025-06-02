import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ==============================================
# VERİ YÜKLEME VE TEMİZLEME
# ==============================================

# Excel dosyasından veriyi yüklüyoruz
df = pd.read_excel("dataset for mendeley 181220.xlsx")

# Sütun isimlerindeki gereksiz boşlukları ve tırnak işaretlerini temizliyoruz
df.columns = df.columns.str.strip().str.replace("'", "", regex=False).str.replace('"', "", regex=False)

# Analizde kullanacağımız sütunları belirliyoruz
required_columns = [
    'Gender',
    'Age as of Academic Year 17/18',
    'Previous Curriculum (17/18)2',
    'Math20-1 ', 'Science20-1 ', 'English20-1 ',
    'Math20-2 ', 'Science20-2 ', 'English20-2 ',
    'Math20-3 ', 'Science20-3 ', 'English20-3 '
]

# Sadece gerekli sütunları alıp, eksik değerleri çıkarıyoruz
df = df[required_columns].dropna()

# Sınav sütunlarını belirliyoruz (Math, Science, English içeren sütunlar)
exam_cols = [col for col in df.columns if "Math" in col or "Science" in col or "English" in col]

# Tüm sınavların ortalamasını hesaplıyoruz
df['ExamAverage'] = df[exam_cols].mean(axis=1)

# Yerleştirme seviyesini belirleyen fonksiyon tanımlıyoruz
def get_placement_level(avg):
    """
    Sınav ortalamasına göre yerleştirme seviyesini belirler
    85+ : High (Yüksek)
    75-84: Medium (Orta)  
    75 altı: Low (Düşük)
    """
    if avg >= 85:
        return 'High'
    elif avg >= 75:
        return 'Medium'
    else:
        return 'Low'

# Her öğrenci için yerleştirme seviyesini hesaplıyoruz
df['PlacementLevel'] = df['ExamAverage'].apply(get_placement_level)

# ==============================================
# VERİ MADENCİLİĞİ GÖRSELLEŞTİRME (EDA)
# ==============================================

# Yerleştirme seviyelerinin dağılımını görselleştiriyoruz
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='PlacementLevel', order=['Low', 'Medium', 'High'], palette='Set2')
plt.title("PlacementLevel Sınıf Dağılımı")
plt.tight_layout()
plt.show()

# Sınav ortalamalarının histogramını çiziyoruz
plt.figure(figsize=(6, 4))
sns.histplot(df['ExamAverage'], kde=True, bins=20, color='skyblue')
plt.title("ExamAverage Dağılımı")
plt.tight_layout()
plt.show()

# Cinsiyete göre sınav ortalamalarını karşılaştırıyoruz
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x='Gender', y='ExamAverage', palette='pastel')
plt.title("Cinsiyete Göre Ortalama Notlar")
plt.tight_layout()
plt.show()

# Önceki müfredata göre sınav ortalamalarını karşılaştırıyoruz
plt.figure(figsize=(8, 4))
sns.boxplot(data=df, x='Previous Curriculum (17/18)2', y='ExamAverage', palette='muted')
plt.title("Müfredata Göre Ortalama Notlar")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Sayısal değişkenler arasındaki korelasyonu inceliyoruz
plt.figure(figsize=(10, 8))
corr = df.select_dtypes(include='number').corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title("Sayısal Özellikler Arası Korelasyon Matrisi")
plt.tight_layout()
plt.show()

# ==============================================
# MODELLEME – RANDOM FOREST
# ==============================================

# Bağımsız değişkenler (X) ve hedef değişken (y) ayırıyoruz
X = df.drop(['ExamAverage', 'PlacementLevel'], axis=1)  # ExamAverage ve PlacementLevel'ı çıkarıyoruz
y = df['PlacementLevel']  # Tahmin etmek istediğimiz değişken

# Kategorik değişkenleri one-hot encoding değişkenlere çeviriyoruz
X = pd.get_dummies(X, columns=['Gender', 'Previous Curriculum (17/18)2'], drop_first=True)

# Veriyi eğitim (%70), doğrulama (%10) ve test (%20) olarak ayırıyoruz
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=2/3, random_state=42, stratify=y_temp)

# Sınıf isimlerini alıyoruz (High, Low, Medium)
class_names = np.unique(y)

# ===============================
# EN İYİ RFC PARAMETRELERİNİ BUL (GRID SEARCH)
# ===============================

# Random Forest için deneyeceğimiz parametre kombinasyonlarını belirliyoruz
param_grid = {
    'n_estimators': [50, 100, 200],        # Ağaç sayısı
    'max_depth': [5, 10, 15],              # Maksimum derinlik
    'min_samples_split': [2, 5],           # Dallanma için minimum örnek sayısı
    'min_samples_leaf': [1, 2],            # Yaprakta minimum örnek sayısı
    'max_features': ['sqrt', 'log2'],       # Her dallanmada kullanılacak özellik sayısı
    'criterion': ['gini', 'entropy']       # Dallanma kriteri
}

# Grid Search ile en iyi parametreleri buluyoruz
print("Grid Search başlıyor... Bu işlem biraz zaman alabilir.")
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(bootstrap=True, random_state=42, n_jobs=-1),
    param_grid=param_grid,
    scoring='f1_macro',  # F1 skoruna göre değerlendirme yapıyoruz
    cv=3,               # 3-fold cross validation
    verbose=2,          # İlerleme durumunu göster
    n_jobs=-1          # Tüm işlemci çekirdeklerini kullan
)

# En iyi parametreleri bulmak için Grid Search çalıştırıyoruz
grid_search.fit(X_train, y_train)

print("\nEn iyi parametreler (GridSearchCV ile):")
best_params = grid_search.best_params_
print(best_params)

# ===============================
# EN İYİ MODEL İLE EĞİTİM VE TEST
# ===============================

# En iyi parametrelerle bulunan modeli alıyoruz
best_model = grid_search.best_estimator_

# Modeli eğitim verisiyle eğitiyoruz
print("En iyi model eğitiliyor...")
best_model.fit(X_train, y_train)

# Test verisi üzerinde tahmin yapıyoruz
y_pred = best_model.predict(X_test)              # Sınıf tahminleri
y_prob = best_model.predict_proba(X_test)        # Olasılık tahminleri

# Test verisi için binary format (ROC eğrisi için gerekli)
y_test_bin = label_binarize(y_test, classes=class_names)

# Eğitim verisi üzerinde de tahmin yapıyoruz (overfitting kontrolü için)
y_train_pred = best_model.predict(X_train)
y_train_prob = best_model.predict_proba(X_train)
y_train_bin = label_binarize(y_train, classes=class_names)

# Temel performans metriklerini hesaplıyoruz
acc = accuracy_score(y_test, y_pred)                              # Doğruluk
prec = precision_score(y_test, y_pred, average='macro')           # Kesinlik
recall = recall_score(y_test, y_pred, average='macro')            # Duyarlılık (Sensitivity)
f1 = f1_score(y_test, y_pred, average='macro')                   # F1 Skoru
auc = roc_auc_score(y_test_bin, y_prob, average='macro', multi_class='ovr')  # AUC

# Specificity (Özgüllük) hesaplıyoruz - her sınıf için ayrı ayrı
specificities = []
for label in class_names:
    # Her sınıf için binary classification problemi haline getiriyoruz
    binary_y_test = (y_test == label).astype(int)
    binary_y_pred = (y_pred == label).astype(int)
    
    # Confusion matrix elemanlarını alıyoruz
    tn, fp, fn, tp = confusion_matrix(binary_y_test, binary_y_pred).ravel()
    
    # Specificity = TN / (TN + FP)
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    specificities.append(specificity)

# Ortalama specificity hesaplıyoruz
specificity_macro = np.mean(specificities)

# Sonuçları ekrana yazdırıyoruz
print("\nEn iyi model test sonuçları:")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall (Sensitivity): {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"AUC: {auc:.4f}")
print(f"Specificity: {specificity_macro:.4f}")

# Confusion Matrix (Karışıklık Matrisi) görselleştiriyoruz
cm = confusion_matrix(y_test, y_pred, labels=class_names)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Greens')
plt.title("En İyi Model - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ROC Eğrilerini çiziyoruz (hem eğitim hem de test için)
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Her sınıf için ROC eğrisini çiziyoruz
for i, class_label in enumerate(class_names):
    # Test verisi için ROC eğrisi
    fpr_test, tpr_test, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    # Eğitim verisi için ROC eğrisi (overfitting kontrolü)
    fpr_train, tpr_train, _ = roc_curve(y_train_bin[:, i], y_train_prob[:, i])
    
    # Eğitim ROC eğrisini çiziyoruz
    axs[0].plot(fpr_train, tpr_train, 
               label=f'{class_label} (Train AUC={roc_auc_score(y_train_bin[:, i], y_train_prob[:, i]):.2f})')
    
    # Test ROC eğrisini çiziyoruz
    axs[1].plot(fpr_test, tpr_test, 
               label=f'{class_label} (Test AUC={roc_auc_score(y_test_bin[:, i], y_prob[:, i]):.2f})')

# Rastgele tahmin çizgisini ekliyoruz (referans için)
axs[0].plot([0, 1], [0, 1], 'k--')
axs[1].plot([0, 1], [0, 1], 'k--')

# Grafik başlıklarını ve etiketlerini ayarlıyoruz
axs[0].set_title("Train ROC Curve (Best RFC)")
axs[1].set_title("Test ROC Curve (Best RFC)")

for ax in axs:
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()

plt.tight_layout()
plt.show()

# ==============================================
# PERFORMANS METRİKLERİ GÖRSELLEŞTİRME
# ==============================================

# Tüm performans metriklerini bir sözlükte topluyoruz
metrics = {
    "Accuracy": acc,
    "Precision": prec,
    "Recall (Sensitivity)": recall,
    "F1-Score": f1,
    "AUC": auc,
    "Specificity": specificity_macro
}

# Performans metriklerini bar grafik olarak görselleştiriyoruz
plt.figure(figsize=(8, 5))
sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette='viridis')
plt.ylim(0, 1.05)  # Y ekseni sınırlarını 0-1 arası yapıyoruz
plt.ylabel("Skor")
plt.title("Model Performans Metrikleri (Test Verisi)")

# Her çubuk üzerine değeri yazıyoruz
for i, v in enumerate(metrics.values()):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center', va='bottom', fontweight='bold')

plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# ==============================================
# ÖZNİTELİK ÖNEMİ GÖRSELLEŞTİRME (FEATURE IMPORTANCE)
# ==============================================

# Random Forest modelinden öznitelik önem skorlarını alıyoruz
importances = best_model.feature_importances_
feature_names = X.columns

# Öznitelik önem skorlarını DataFrame'e çevirip sıralıyoruz
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# En önemli öznitelikleri görselleştiriyoruz
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance_df.head(50), x='Importance', y='Feature', palette='mako')
plt.title("Öznitelik Önem Sıralamaları (Random Forest)")
plt.xlabel("Önem Skoru")
plt.ylabel("Öznitelik")
plt.tight_layout()
plt.show()

print("\nAnaliz tamamlandı! Model başarıyla eğitildi ve değerlendirildi.")
print(f"En yüksek performans metrikleri:")
print(f"- En yüksek skor: {max(metrics.values()):.4f}")
print(f"- Bu skorun ait olduğu metrik: {max(metrics, key=metrics.get)}")