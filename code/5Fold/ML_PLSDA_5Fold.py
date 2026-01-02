import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.cross_decomposition import PLSRegression
from collections import defaultdict
import os
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from catboost import CatBoostClassifier, CatBoostError
import matplotlib;
matplotlib.use("Agg")
import matplotlib.pyplot as plt;
plt.ioff()
from catboost.utils import get_gpu_device_count
_cat_task = "GPU" if get_gpu_device_count() > 0 else "CPU"

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
    y = pd.read_parquet(y_path)['label'].values

    # --- Teyit Kontrolleri ---
    y_sr = pd.read_parquet(y_path)['label']
    print("Sınıf dağılımı:\n", y_sr.value_counts())
    assert set(y_sr.unique()) == {0, 1}, "Etiketler 0/1 değil!"

    print("X shape:", X.shape, "| y shape:", y.shape)
    assert len(X) == len(y), "X ve y uzunlukları uyuşmuyor!"
    assert np.isfinite(X).all(), "X içinde NaN/Inf var!"

    selector = VarianceThreshold(threshold=0.0)
    X = selector.fit_transform(X)
    print(f"Sabit Sütunlar Kaldırıldıktan Sonraki Sütun Sayısı: {X.shape[1]}")

except Exception as e:
    print(f"BÖLÜM 2'DE KRİTİK HATA OLUŞTU: {e}")
    raise RuntimeError(f"Veri yükleme/ön-işleme hatası: {e}") from e


def bulletproof_clean(data):
    finfo32 = np.finfo(np.float32)
    clean_data = np.asarray(data)
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


best_params_store = {
    "GradientBoost": {
        "n_estimators": 90,
        "learning_rate": 0.1,
        "max_depth": 6,
        "subsample": 0.8,
        "max_features": "sqrt",
        "random_state": 42,
    },
    "HistGB": {
        "max_iter": 100,
        "learning_rate": 0.1,
        "l2_regularization": 1.0,
    },
    "CatBoost": {
        "iterations": 300,
        "learning_rate": 0.1,
        "depth": 8,
    },
    "SGDClassifier": {
        "loss": "log_loss",
        "penalty": "elasticnet",
        "alpha": 1e-4,
        "l1_ratio": 0.15,
        "max_iter": 2000,
        "tol": 1e-3,
        "class_weight": "balanced",
        "early_stopping": True,
        "n_iter_no_change": 8,
        "learning_rate": "adaptive",
        "eta0": 0.01,
        "random_state": 42
    },

    "ExtraTrees": {
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "bootstrap": False,
        "random_state": 42,
        "n_jobs": -1
    }
}
# BÖLÜM 3: ANA DEĞERLENDİRME DÖNGÜSÜ

assert X.ndim == 2 and y.ndim == 1 and len(X) == len(y), "X/y boyutları uyuşmuyor."
cm_totals = defaultdict(lambda: np.zeros((2, 2), dtype=int))

print("\nBÖLÜM 3: 5-Katlı Çapraz Doğrulama ile Değerlendirme Başlatılıyor...")
all_results = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_index, test_index) in enumerate(skf.split(X, y), start=1):
    print(f"\n--- FOLD {fold}/5 ---")

    # Split
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Her adımdan sonra temizlik ile sağlamlaştırılmış boru hattı
    scaler = RobustScaler()
    X_train_scaled = bulletproof_clean(scaler.fit_transform(X_train))
    X_test_scaled = bulletproof_clean(scaler.transform(X_test))

    vt = VarianceThreshold(0.0)
    X_train_scaled = vt.fit_transform(X_train_scaled)
    X_test_scaled = vt.transform(X_test_scaled)

    pls = PLSRegression(
        n_components=64,  # 2500+ özellik için uygun bir sıkıştırma
        scale=False,  # RobustScaler sonrası ekstra ölçekleme yapma
        max_iter=1000,  # yakınsama için güvenli üst sınır
        tol=1e-06
    )

    #  fit_transform çıktısının ilk elemanı X_scores olduğu için [0] ekliy
    X_train_pls = bulletproof_clean(pls.fit_transform(X_train_scaled, y_train.astype(float))[0])
    X_test_pls = bulletproof_clean(pls.transform(X_test_scaled))
    print(f"PLS dtypes -> train: {X_train_pls.dtype}, test: {X_test_pls.dtype}")

    datasets = {
        "PLS-DA": (X_train_pls, y_train, X_test_pls),
    }

    for veri_name, (X_train_curr, y_train_curr, X_test_curr) in datasets.items():
        print(f"Fold {fold} - Veri Tipi: {veri_name} - Modeller çalıştırılıyor...")

        # 1️ Bulletproof temizlik
        X_train_final = bulletproof_clean(X_train_curr)
        X_test_final = bulletproof_clean(X_test_curr)

        # 2️ NaN/Inf temizliği (her veri tipi için)
        X_train_final = np.nan_to_num(X_train_final, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_final = np.nan_to_num(X_test_final, nan=0.0, posinf=0.0, neginf=0.0)

        ml_models = {
            "GradientBoost": GradientBoostingClassifier(**best_params_store["GradientBoost"]),
            "HistGB": HistGradientBoostingClassifier(**best_params_store["HistGB"]),
            "CatBoost": CatBoostClassifier(
                **best_params_store["CatBoost"],
                task_type=_cat_task, devices="0",
                verbose=0, allow_writing_files=False
            ),
            "SGDClassifier": SGDClassifier(**best_params_store["SGDClassifier"]),
            "ExtraTrees": ExtraTreesClassifier(**best_params_store["ExtraTrees"]),
        }

        for name, model in ml_models.items():

            try:
                # ---- TRAIN ----
                model.fit(X_train_final, y_train_curr)

                # ---- TEST ----
                y_pred_test = model.predict(X_test_final)

                # ---- TEST PROBA  ----
                if hasattr(model, "predict_proba"):
                    y_prob_test = model.predict_proba(X_test_final)[:, 1]
                elif hasattr(model, "decision_function"):
                    z = model.decision_function(X_test_final)
                    y_prob_test = 1.0 / (1.0 + np.exp(-z))
                else:
                    y_prob_test = model.predict(X_test_final).astype(float)

                add_results(y_test, y_pred_test, y_prob_test, name, "Test", veri_name, fold, all_results)
                print(f"Kaydedildi -> Veri:{veri_name}, Model:{name}, Set:Test")

                #  TOPLAMA
                cm = confusion_matrix(y_test, y_pred_test, labels=[0, 1])
                cm_totals[("Test", veri_name, name)] += cm

                y_pred_train = model.predict(X_train_final)

                # ---- TRAIN PROBA ----
                if hasattr(model, "predict_proba"):
                    y_prob_train = model.predict_proba(X_train_final)[:, 1]
                elif hasattr(model, "decision_function"):
                    zt = model.decision_function(X_train_final)
                    y_prob_train = 1.0 / (1.0 + np.exp(-zt))
                else:
                    y_prob_train = model.predict(X_train_final).astype(float)

                add_results(y_train_curr, y_pred_train, y_prob_train, name, "Train", veri_name, fold, all_results)
                print(f"Kaydedildi -> Veri:{veri_name}, Model:{name}, Set:Train")



            except CatBoostError as e:
                print(f"UYARI: {name} modeli '{veri_name}' verisiyle eğitilemedi. Hata: {e}. Bu model atlanıyor.")
            except Exception as e:
                print(f"BEKLENMEDİK HATA: {name} modeli '{veri_name}' verisiyle çalışırken çöktü. Hata: {e}")

# BÖLÜM 4: Sonuçların Raporlanması (sadece kullanılan metriklerle
if 'all_results' in locals() and all_results:
    print("\n\nBÖLÜM 4: Tüm Değerlendirme Tamamlandı. Sonuçlar Raporlanıyor...")

    # Masaüstünde "EMBER RESULT" klasörü
    save_dir = os.path.join(os.path.expanduser("~"), "Desktop", "EMBER RESULT")
    os.makedirs(save_dir, exist_ok=True)

    # --- Nihai (Toplam) Karışıklık Matrisleri Kaydediliyor ---
    print("\n--- Nihai (Toplam) Karışıklık Matrisleri Kaydediliyor ---")
    for (set_name, vname, mname), cm in cm_totals.items():
        try:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Malicious'])
            disp.plot(cmap="Blues", values_format="d", colorbar=False)
            plt.title(f"Total Confusion Matrix\nModel: {mname} | Veri: {vname} | Set: {set_name}")
            fig = plt.gcf()
            fig.tight_layout()
            save_path = os.path.join(save_dir, f"TOTAL_{vname}_{mname}_CM.png")
            fig.savefig(save_path, dpi=300)
            plt.close(fig)
            print(f"Kaydedildi → {save_path}")
        except Exception as e:
            print(f"[UYARI] CM çiziminde hata ({vname}-{mname}): {e}")

    # --- Sonuç tablosu ---
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
    csv_path = os.path.join(save_dir, "EMBER24_final_model_performance_summary_FULL.csv")
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


