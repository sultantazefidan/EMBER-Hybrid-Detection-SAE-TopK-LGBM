import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)

# --- Genel Kayıt Klasörü ---
save_dir = os.path.join(os.path.expanduser("~"), "Desktop", "EMBER RESULT")
os.makedirs(save_dir, exist_ok=True)

try:
    print("veri yukleniyor")

    # --- Dosya Yolları ---
    X_path = r"C:\Users\Gaming\Desktop\EMBER24_400k.Main_Vec_Cleann\parquet\X_train_varcorr_clean.parquet"
    y_path = r"C:\Users\Gaming\Desktop\EMBER24_400k.Main_Vec_Cleann\parquet\y_train_clean.parquet"

    # --- Yükleme ---
    X = pd.read_parquet(X_path).values
    y = pd.read_parquet(y_path)['label'].values  # etiket sütunu: 'label'

    # --- Teyit Kontrolleri ---
    y_sr = pd.read_parquet(y_path)['label']
    print("Sınıf dağılımı:\n", y_sr.value_counts())
    assert set(y_sr.unique()) == {0, 1}, "Etiketler 0/1 değil!"


    print("X shape:", X.shape, "| y shape:", y.shape)
    assert len(X) == len(y), "X ve y uzunlukları uyuşmuyor!"
    assert np.isfinite(X).all(), "X içinde NaN/Inf var!"

    # Sabit sütunları kaldır
    selector = VarianceThreshold(threshold=0.0)
    X = selector.fit_transform(X)
    print(f"Sabit Sütunlar Kaldırıldıktan Sonraki Sütun Sayısı: {X.shape[1]}")

except Exception as e:
    print(f"BÖLÜM 2'DE KRİTİK HATA OLUŞTU: {e}")
    raise RuntimeError(f"Veri yükleme/ön-işleme hatası: {e}") from e


def bulletproof_clean(data):
    finfo32 = np.finfo(np.float32)
    clean_data = np.asarray(data)  # dtype float 32 olarak kalsn
    clean_data = np.nan_to_num(clean_data, nan=0.0,
                               posinf=finfo32.max, neginf=finfo32.min)
    return np.clip(clean_data, finfo32.min, finfo32.max)


def add_results(y_true, y_pred, y_prob, model_name, set_name, veri_name, fold, results_list):
    y_prob_clean = bulletproof_clean(np.asarray(y_prob).ravel())
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    results_list.append({
        "Fold": fold,
        "Veri": veri_name,
        "Model": model_name,
        "Set": set_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": specificity,
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "F1 Weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "AUC-ROC": roc_auc_score(y_true, y_prob_clean),
    })


#  BÖLÜM 3: 5-Katlı Çapraz Doğrulama ile Değerlendirme
assert X.ndim == 2 and y.ndim == 1 and len(X) == len(y), "X/y boyutları uyuşmuyor."

from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from lightgbm import LGBMClassifier
import numpy as np, os, pandas as pd, matplotlib.pyplot as plt

cm_totals = defaultdict(lambda: np.zeros((2, 2), dtype=int))
all_results = []

print("\nBÖLÜM 3: 5-Katlı Çapraz Doğrulama ile Değerlendirme Başlatılıyor...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_index, test_index) in enumerate(skf.split(X, y), start=1):
    print(f"\n--- FOLD {fold + 1}/5 ---")

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Her adımdan sonra temizlik ile sağlamlaştırılmış boru hattı
    scaler = RobustScaler()
    X_train_scaled = bulletproof_clean(scaler.fit_transform(X_train))
    X_test_scaled  = bulletproof_clean(scaler.transform(X_test))

    vt = VarianceThreshold(0.0)
    X_train_scaled = vt.fit_transform(X_train_scaled)
    X_test_scaled  = vt.transform(X_test_scaled)


    datasets = {
        "OFF": (X_train_scaled, y_train, X_test_scaled),
    }

    #  Sadece LGBM parametreleri
    ml_models = {
        "LGBM": LGBMClassifier(
            objective="binary",
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=50,
            max_depth=-1,
            reg_lambda=1.0,
            min_child_samples=50,
            random_state=42
        )
    }


    for veri_name, (X_train_curr, y_train_curr, X_test_curr) in datasets.items():
        print(f"Fold {fold+1} - Veri Tipi: {veri_name} - lgbm çalıştırılıyor...")

        X_train_final = bulletproof_clean(X_train_curr)
        X_test_final  = bulletproof_clean(X_test_curr)

        X_train_final = np.nan_to_num(X_train_final, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_final  = np.nan_to_num(X_test_final,  nan=0.0, posinf=0.0, neginf=0.0)

        for name, model in ml_models.items():
            try:
                model.fit(X_train_final, y_train_curr)

                y_pred_test = model.predict(X_test_final)
                y_prob_test = model.predict_proba(X_test_final)[:, 1]
                add_results(y_test, y_pred_test, y_prob_test, name, "Test", veri_name, fold, all_results)
                print(f"Kaydedildi -> Veri:{veri_name}, Model:{name}, Set:Test")

                cm = confusion_matrix(y_test, y_pred_test, labels=[0, 1])
                cm_totals[("Test", veri_name, name)] += cm

                y_pred_train = model.predict(X_train_final)
                y_prob_train = model.predict_proba(X_train_final)[:, 1]
                add_results(y_train_curr, y_pred_train, y_prob_train, name, "Train", veri_name, fold, all_results)

            except Exception as e:
                print(f"[HATA][Fold {fold}][{veri_name}][{name}] -> {e}")

#BÖLÜM 4: Sonuçların Raporlanması
if all_results:
    print("\n\nBÖLÜM 4: Tüm Değerlendirme Tamamlandı. Sonuçlar Raporlanıyor...")

    # Masaüstünde "EMBER RESULT" klasörü
    save_dir = os.path.join(os.path.expanduser("~"), "Desktop", "EMBER RESULT")
    os.makedirs(save_dir, exist_ok=True)

    #  TOPLAM CM'leri kaydet
    for (set_name, vname, mname), cm in cm_totals.items():
        try:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Malicious'])
            disp.plot()
            fig = plt.gcf()
            fig.tight_layout()
            fig.savefig(os.path.join(save_dir, f'TOTAL_{vname}_{mname}_CM_{set_name}.png'))
            plt.close(fig)
        except Exception as e:
            print(f"[Uyarı] CM kaydedilemedi ({vname}-{mname}-{set_name}): {e}")

    # Sonuç tablosu
    df_results = (
        pd.DataFrame(all_results)
          .replace([np.inf, -np.inf], np.nan)
          .dropna(how='any')
    )

    final_summary = (
        df_results
        .groupby(['Veri', 'Model', 'Set'], as_index=True)
        .agg(
            Accuracy_mean=('Accuracy', 'mean'), Accuracy_std=('Accuracy', 'std'),
            Precision_mean=('Precision', 'mean'), Precision_std=('Precision', 'std'),
            Recall_mean=('Recall', 'mean'), Recall_std=('Recall', 'std'),
            Specificity_mean=('Specificity', 'mean'), Specificity_std=('Specificity', 'std'),
            F1_mean=('F1 Score', 'mean'), F1_std=('F1 Score', 'std'),
            F1W_mean=('F1 Weighted', 'mean'), F1W_std=('F1 Weighted', 'std'),
            AUC_ROC_mean=('AUC-ROC', 'mean'), AUC_ROC_std=('AUC-ROC', 'std'),
        )
        .round(4)
        .sort_index()
    )

    # Kaydetme yolları
    csv_path  = os.path.join(save_dir, "EMBER24_final_model_performance_summary_FULL.csv")
    xlsx_path = os.path.join(save_dir, "EMBER24_final_model_performance_summary_FULL.xlsx")

    final_summary.to_csv(csv_path)
    final_summary.to_excel(xlsx_path)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 120)
    print("\n--- Nihai Model Performans Özeti (Ortalama ± Std. Sapma) ---")
    print(final_summary)
    print("\nDosyalar kaydedildi:\n", csv_path, "\n", xlsx_path)
else:
    print("\n\nBÖLÜM 4: Raporlanacak sonuç bulunamadı.")



