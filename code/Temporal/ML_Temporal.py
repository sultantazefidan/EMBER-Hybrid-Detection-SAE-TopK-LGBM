import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib;

matplotlib.use("Agg")
import matplotlib.pyplot as plt;

plt.ioff()

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import ExtraTreesClassifier

from catboost import CatBoostClassifier, CatBoostError


#Yardımcılar
def bulletproof_clean(data):
    finfo32 = np.finfo(np.float32)
    arr = np.asarray(data, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=finfo32.max, neginf=finfo32.min)
    return np.clip(arr, finfo32.min, finfo32.max)


def add_results(y_true, y_pred, y_prob, model_name, results_list):
    y_prob_clean = bulletproof_clean(np.asarray(y_prob).ravel())
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    results_list.append({
        "Veri": "OFF", "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": specificity,
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "F1 Weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "AUC-ROC": roc_auc_score(y_true, y_prob_clean),
    })


#Kayıt klasörü
save_dir = os.path.join(os.path.expanduser("~"), "Desktop", "EMBER RESULT")
os.makedirs(save_dir, exist_ok=True)

#Veri Yolları
train_X_path = r"C:\Users\Gaming\Desktop\EMBER24_200k.Main_Vec_Cleann\parquet\X_train_varcorr_clean.parquet"
train_y_path = r"C:\Users\Gaming\Desktop\EMBER24_200k.Main_Vec_Cleann\parquet\y_train.parquet"
test_X_path = r"C:\Users\Gaming\Desktop\EMBER24_200K.temporal_Vec_Clean\parquet\X_train_varcorr_clean.parquet"
test_y_path = r"C:\Users\Gaming\Desktop\EMBER24_200K.temporal_Vec_Clean\parquet\y_test_clean.parquet"

OUT_X_ALIGNED = os.path.join(save_dir, "X_test_temporal_aligned.parquet")
OUT_LOG = os.path.join(save_dir, "feature_alignment_log.txt")

print("Veri yükleniyor...")
X_train_df = pd.read_parquet(train_X_path).astype(np.float32)
y_train_df = pd.read_parquet(train_y_path)
X_test_df = pd.read_parquet(test_X_path).astype(np.float32)
y_test_df = pd.read_parquet(test_y_path)
print(f"X_train shape: {X_train_df.shape} | X_test (raw): {X_test_df.shape}")

#Kolon farkları + hizalama + kontroller + log 
train_cols = X_train_df.columns.tolist()
test_cols = X_test_df.columns.tolist()

missing_in_test = [c for c in train_cols if c not in test_cols]
extra_in_test = [c for c in test_cols if c not in train_cols]

print(f"- Eğitimde olup testte eksik sütun: {len(missing_in_test)}")
print(f"- Testte olup eğitimde olmayan sütun: {len(extra_in_test)}")

#Eksikleri 0 ile ekle, fazlaları at, sıralamayı hizala
for c in missing_in_test:
    X_test_df[c] = np.float32(0.0)
if extra_in_test:
    X_test_df = X_test_df.drop(columns=extra_in_test, errors="ignore")
X_test_df = X_test_df[train_cols]

#Son kontroller
assert X_test_df.shape[1] == X_train_df.shape[1], "Kolon sayısı eşleşmedi!"
assert list(X_test_df.columns) == train_cols, "Kolon sıralaması farklı!"
assert np.isfinite(X_test_df.to_numpy()).all(), "X_test içinde NaN/Inf var!"

# Hizalanmış test X'i kaydet + log
pd.DataFrame(X_test_df, columns=train_cols).to_parquet(OUT_X_ALIGNED, index=False)
log_lines = [
    f"Train shape: {X_train_df.shape}",
    f"Test raw shape: {pd.read_parquet(test_X_path).shape}",
    f"Missing in test ({len(missing_in_test)}): {missing_in_test}",
    f"Extra in test   ({len(extra_in_test)}): {extra_in_test}",
    f"Aligned test shape: {X_test_df.shape}",
    f"Saved: {OUT_X_ALIGNED}",
]
Path(OUT_LOG).write_text("\n".join(log_lines), encoding="utf-8")
print(f"Log: {OUT_LOG}")

#y dizileri
y_train = (y_train_df['label'] if 'label' in y_train_df.columns else y_train_df.squeeze()).to_numpy().astype(int)
y_test = (y_test_df['label'] if 'label' in y_test_df.columns else y_test_df.squeeze()).to_numpy().astype(int)


scaler = StandardScaler()
X_train = bulletproof_clean(scaler.fit_transform(X_train_df))
X_test = bulletproof_clean(scaler.transform(X_test_df))

vt = VarianceThreshold(0.0)
X_train = bulletproof_clean(vt.fit_transform(X_train))
X_test = bulletproof_clean(vt.transform(X_test))
print(f"[CHECK OFF] after scaler+VT -> train: {X_train.shape}, test: {X_test.shape}")

#Modeller (yalnız CatBoost & ExtraTrees) 
cat_params = dict(
    iterations=150,
    learning_rate=0.3,
    depth=4,
    l2_leaf_reg=10,
    verbose=0, allow_writing_files=False

)

ml_models = {
    "CatBoost": CatBoostClassifier(**cat_params),

    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features=0.5,
        bootstrap=True,
        random_state=42, n_jobs=-1
    )
}

#Değerlendirme
cm_totals = defaultdict(lambda: np.zeros((2, 2), dtype=int))
all_results = []

print("\n=== TEMPORAL SPLIT: OFF (CatBoost & ExtraTrees) ===")
for name, model in ml_models.items():
    try:
        model.fit(X_train, y_train)

        #TEST
        y_pred_test = model.predict(X_test)
        y_prob_test = model.predict_proba(X_test)[:, 1]

        add_results(y_test, y_pred_test, y_prob_test, name, all_results)
        cm_totals[name] += confusion_matrix(y_test, y_pred_test, labels=[0, 1])

        #Konsol özeti
        acc = accuracy_score(y_test, y_pred_test)
        prec = precision_score(y_test, y_pred_test, zero_division=0)
        rec = recall_score(y_test, y_pred_test, zero_division=0)
        f1 = f1_score(y_test, y_pred_test, zero_division=0)
        auc = roc_auc_score(y_test, bulletproof_clean(y_prob_test))
        print(f"{name:10s} | Acc={acc:.4f}  Prec={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}  AUC={auc:.4f}")

    except CatBoostError as e:
        print(f"[UYARI] CatBoost hata: {e} (atlanıyor).")
    except Exception as e:
        print(f"[HATA] {name} çalışırken: {e}")

#Raporlama
if all_results:
    # Toplam CM görselleri
    for mname, cm in cm_totals.items():
        try:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Malicious'])
            disp.plot(cmap="Blues", values_format="d", colorbar=False)
            plt.title(f"Total Confusion Matrix\nModel: {mname} | Veri: OFF")
            fig = plt.gcf();
            fig.tight_layout()
            out_cm = os.path.join(save_dir, f"TEMPORAL_TOTAL_OFF_{mname}_CM.png")
            fig.savefig(out_cm, dpi=300);
            plt.close(fig)
            print(f"CM kaydedildi → {out_cm}")
        except Exception as e:
            print(f"[UYARI] CM çizim hatası ({mname}): {e}")

    # Özet tablo
    df = (pd.DataFrame(all_results)
          .replace([np.inf, -np.inf], np.nan)
          .dropna(how="any"))
    summary = (df.groupby(['Veri', 'Model'], as_index=True)
               .agg(Accuracy=('Accuracy', 'mean'),
                    Precision=('Precision', 'mean'),
                    Recall=('Recall', 'mean'),
                    Specificity=('Specificity', 'mean'),
                    F1=('F1 Score', 'mean'),
                    F1W=('F1 Weighted', 'mean'),
                    AUC_ROC=('AUC-ROC', 'mean'))
               .round(6)
               .sort_index())
    csv_path = os.path.join(save_dir, "EMBER24_TEMPORAL_OFF_performance_summary.csv")
    xlsx_path = os.path.join(save_dir, "EMBER24_TEMPORAL_OFF_performance_summary.xlsx")
    summary.to_csv(csv_path);
    summary.to_excel(xlsx_path)
    print("\nÖzet tablo kaydedildi:\n", csv_path, "\n", xlsx_path)
else:
    print("Sonuç yok.")


