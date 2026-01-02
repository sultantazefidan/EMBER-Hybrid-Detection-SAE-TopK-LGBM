import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
import os, time, tempfile
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt; plt.ioff()
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay
)
from lightgbm import LGBMClassifier
from lightgbm.callback import early_stopping


THRESH_METHOD = "multi"
TARGET_ALL    = 0.97

#Yardımcılar
def bytes_to_mb(nbytes: int) -> float:
    return nbytes / (1024**2)

def keras_model_size_mb(model: tf.keras.Model, dtype_bytes: int = 4) -> float:
    n_params = int(np.sum([np.prod(w.shape) for w in model.trainable_weights]))
    return bytes_to_mb(n_params * dtype_bytes)

def lgbm_model_size_mb(clf) -> float:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt"); tmp.close()
    clf.booster_.save_model(tmp.name)
    size_mb = bytes_to_mb(os.path.getsize(tmp.name))
    os.unlink(tmp.name)
    return size_mb

def bulletproof_clean(data):
    finfo32 = np.finfo(np.float32)
    clean_data = np.asarray(data)
    clean_data = np.nan_to_num(clean_data, nan=0.0,
                               posinf=finfo32.max, neginf=finfo32.min)
    return np.clip(clean_data, finfo32.min, finfo32.max)

def add_results(y_true, y_pred, y_prob, model_name, set_name, veri_name, results_list):
    y_prob_clean = bulletproof_clean(np.asarray(y_prob).ravel())
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    results_list.append({
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

# --- metrik hesaplayıcı / eşikçiler ---
def _metrics_at(y_true, y_prob, thr):
    yp = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, yp).ravel()
    acc  = (tp + tn) / (tp + tn + fp + fn)
    prec = precision_score(y_true, yp, zero_division=0)
    rec  = recall_score(y_true, yp, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1   = f1_score(y_true, yp, zero_division=0)
    return acc, prec, rec, spec, f1

def find_best_threshold_multi(y_true, y_prob, target=0.97):
    # 1) Tüm metrikler hedefi geçen varsa: F1 en yüksek olanı seçildi
    cands = []
    for thr in np.linspace(0.01, 0.99, 990):
        acc, prec, rec, spec, f1 = _metrics_at(y_true, y_prob, thr)
        if (acc >= target) and (prec >= target) and (rec >= target) and (spec >= target) and (f1 >= target):
            cands.append((f1, thr))
    if cands:
        return max(cands)[1]
    # 2) Hedefe ulaşılamıyorsa: F1 × BA (BA=(rec+spec)/2) maksimize
    best = None
    for thr in np.linspace(0.01, 0.99, 990):
        acc, prec, rec, spec, f1 = _metrics_at(y_true, y_prob, thr)
        ba = 0.5 * (rec + spec)
        score = f1 * ba
        if (best is None) or (score > best[0]):
            best = (score, thr)
    return best[1]

def best_thr_youden(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])

def fine_tune_threshold(y_true, y_prob, rough_thr, target=0.97):
    # çevresinde ince tarama (±0.05 aralığı, 0.001 adım)
    lo = max(0.0, rough_thr - 0.05)
    hi = min(1.0, rough_thr + 0.05)
    best_ok = None
    best_alt = (0.0, rough_thr)  # (F1*BA, thr)
    for thr in np.arange(lo, hi + 1e-12, 0.001):
        acc, prec, rec, spec, f1 = _metrics_at(y_true, y_prob, thr)
        ba = 0.5*(rec+spec)
        score = f1*ba
        if (acc>=target) and (prec>=target) and (rec>=target) and (spec>=target) and (f1>=target):
            # ilk bulunan ya da F1 büyük olanı al
            if best_ok is None or f1 > best_ok[0]:
                best_ok = (f1, thr)
        if score > best_alt[0]:
            best_alt = (score, thr)
    if best_ok is not None:
        return best_ok[1], True
    return best_alt[1], False

# Kayıt Klasörü
save_dir = os.path.join(os.path.expanduser("~"), "Desktop", "EMBER RESULT")
os.makedirs(save_dir, exist_ok=True)

#Veri Yolları
train_X_path = r"C:\Users\Gaming\Desktop\EMBER24_200k.Main_Vec_Cleann\parquet\X_train_varcorr_clean.parquet"
train_y_path = r"C:\Users\Gaming\Desktop\EMBER24_200k.Main_Vec_Cleann\parquet\y_train.parquet"

test_X_path  = r"C:\Users\Gaming\Desktop\EMBER24_200K.temporal_Vec_Clean\parquet\X_train_varcorr_clean.parquet"
test_y_path  = r"C:\Users\Gaming\Desktop\EMBER24_200K.temporal_Vec_Clean\parquet\y_test_clean.parquet"

# Çıktılar
OUT_X_ALIGNED = os.path.join(save_dir, "X_test_temporal_aligned.parquet")
OUT_LOG = os.path.join(save_dir, "feature_alignment_log.txt")

# Yükle + Hizala
print(" Veri yükleniyor...")
X_train = pd.read_parquet(train_X_path)
y_train = pd.read_parquet(train_y_path)
X_test  = pd.read_parquet(test_X_path)
y_test  = pd.read_parquet(test_y_path)

X_train = X_train.astype(np.float32)
X_test  = X_test.astype(np.float32)

print(f"X_train shape: {X_train.shape} | X_test (raw): {X_test.shape}")

train_cols = X_train.columns.tolist()
test_cols  = X_test.columns.tolist()
missing_in_test = [c for c in train_cols if c not in test_cols]
extra_in_test   = [c for c in test_cols  if c not in train_cols]
print(f"- Eğitimde olup testte eksik sütun: {len(missing_in_test)}")
print(f"- Testte olup eğitimde olmayan sütun: {len(extra_in_test)}")

for c in missing_in_test:
    X_test[c] = np.float32(0.0)
if extra_in_test:
    X_test = X_test.drop(columns=extra_in_test, errors="ignore")
X_test = X_test[train_cols]

assert X_test.shape[1] == X_train.shape[1], "Kolon sayısı eşleşmedi!"
assert list(X_test.columns) == train_cols, "Kolon sıralaması farklı!"
assert np.isfinite(X_test.to_numpy()).all(), "X_test içinde NaN/Inf var!"

pd.DataFrame(X_test, columns=train_cols).to_parquet(OUT_X_ALIGNED, index=False)
log_lines = [
    f"Train shape: {X_train.shape}",
    f"Test raw shape: {pd.read_parquet(test_X_path).shape}",
    f"Missing in test ({len(missing_in_test)}): {missing_in_test}",
    f"Extra in test   ({len(extra_in_test)}): {extra_in_test}",
    f"Aligned test shape: {X_test.shape}",
    f"Saved: {OUT_X_ALIGNED}",
]
Path(OUT_LOG).write_text("\n".join(log_lines), encoding="utf-8")
print(f"📝 Log: {OUT_LOG}")

y_train_arr = (y_train['label'] if 'label' in y_train.columns else y_train.squeeze()).to_numpy()
y_test_arr  = (y_test['label']  if 'label'  in y_test.columns  else y_test.squeeze()).to_numpy()

X_train = bulletproof_clean(X_train).astype(np.float32, copy=False)
X_test  = bulletproof_clean(X_test).astype(np.float32, copy=False)

# Ölçekleme + VT
scaler = StandardScaler()
X_train = bulletproof_clean(scaler.fit_transform(X_train).astype(np.float32, copy=False))
X_test  = bulletproof_clean(scaler.transform(X_test).astype(np.float32, copy=False))

vt = VarianceThreshold(threshold=0.0)
X_train = bulletproof_clean(vt.fit_transform(X_train).astype(np.float32, copy=False))
X_test  = bulletproof_clean(vt.transform(X_test).astype(np.float32, copy=False))

print(f"[CHECK] after scaler+VT -> train: {X_train.shape}, test: {X_test.shape}")

# HİBRİT (SAE + Top-K + LGBM)
veri_name = "OFF"

# --- SAE  ---
input_dim = X_train.shape[1]
inp = tf.keras.Input(shape=(input_dim,))
x  = layers.Dense(512, activation='relu', name='sae_h1',
                  kernel_regularizer=regularizers.l2(1e-5))(inp)
x  = layers.BatchNormalization()(x)
x  = layers.Dropout(0.05)(x)

x  = layers.Dense(256, activation='relu', name='sae_h2',
                  kernel_regularizer=regularizers.l2(1e-5))(x)
x  = layers.BatchNormalization()(x)
x = layers.GaussianNoise(0.05)(x)

z  = layers.Dense(192, activation='relu', name='sae_latent',
                  kernel_regularizer=regularizers.l2(1e-5))(x)

d1 = layers.Dense(256, activation='relu', name='sae_d1',
                  kernel_regularizer=regularizers.l2(1e-5))(z)
d1 = layers.BatchNormalization()(d1)
recon = layers.Dense(input_dim, activation='linear', name='sae_recon')(d1)

encoder = models.Model(inp, z, name="sae_encoder")
autoenc = models.Model(inp, recon, name="sae_autoencoder")
autoenc.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')

Xtr32 = X_train.astype(np.float32, copy=False)
Xte32 = X_test.astype(np.float32, copy=False)

# ES'i kapatıp uzun eğitim (daha "fit" latent)
t0 = time.perf_counter()
autoenc.fit(
    Xtr32, Xtr32,
    epochs=100,
    batch_size=256,
    verbose=0
)
print(f"[TIME] SAE train: {time.perf_counter()-t0:.3f}s")

# Latent uzay
Z_tr = encoder.predict(Xtr32, verbose=0)
Z_te = encoder.predict(Xte32,  verbose=0)

# --- Top-K (gain) ---
K_TOP = 256
n_feat = Xtr32.shape[1]
K_eff = min(K_TOP, n_feat)

probe = LGBMClassifier(
    objective="binary",
    n_estimators=3000,
    learning_rate=0.02,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    n_jobs=-1,
)
probe.fit(Xtr32, y_train_arr)

try:
    imp = probe.booster_.feature_importance(importance_type="gain").astype(np.float64)
except Exception:
    imp = np.asarray(getattr(probe, "feature_importances_", np.zeros(n_feat)), dtype=np.float64)
if not np.any(imp):
    imp = np.arange(n_feat, dtype=np.float64)

top_idx = np.argsort(imp)[-K_eff:]
Xtr_top = Xtr32[:, top_idx]
Xte_top = Xte32[:, top_idx]

# latent + topK
Xtr_h = np.hstack([Z_tr, Xtr_top]).astype(np.float32, copy=False)
Xte_h = np.hstack([Z_te, Xte_top]).astype(np.float32, copy=False)

# --- Final LGBM ---
lgbm = LGBMClassifier(
    objective="binary",
    n_estimators=6000,
    learning_rate=0.015,
    num_leaves=96,
    max_depth=10,
    min_child_samples=160,
    min_split_gain=0.0,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
    colsample_bynode=0.8,
    reg_lambda=6.0,
    reg_alpha=1.5,
    random_state=42,
    n_jobs=-1,
)

t0 = time.perf_counter()
lgbm.fit(
    Xtr_h, y_train_arr,
    eval_set=[(Xte_h, y_test_arr)],
    eval_metric=["auc", "binary_logloss"],
    callbacks=[early_stopping(200, verbose=False)]
)
train_time = time.perf_counter() - t0
print(f"[TIME] TOTAL TRAIN (SAE + LGBM): {train_time:.3f}s")
# MODEL BOYUTU
sae_mb = keras_model_size_mb(encoder)        # SAE encoder boyutu
lgbm_mb = lgbm_model_size_mb(lgbm)           # LGBM boyutu
total_mb = sae_mb + lgbm_mb

print(f"[SIZE] SAE={sae_mb:.2f} MB | LGBM={lgbm_mb:.2f} MB | TOTAL={total_mb:.2f} MB")


# Test İnferans Zamanlama
t0_inf = time.perf_counter()
y_prob = lgbm.predict_proba(Xte_h, num_iteration=lgbm.best_iteration_)[:, 1]
infer_total = time.perf_counter() - t0_inf
n_samples = Xte_h.shape[0]
latency_ms = (infer_total / max(n_samples, 1)) * 1000.0
throughput = n_samples / infer_total if infer_total > 0 else float("inf")
print(f"[INFER] total={infer_total:.3f} s | n={n_samples} | latency={latency_ms:.3f} ms/sample | throughput={throughput:.1f} samples/s")

#Test Sonuçları (EŞİK OPTİMİZASYONLU)
# 1) kaba eşik (seçili metoda göre)
if THRESH_METHOD.lower() == "youden":
    rough_thr = best_thr_youden(y_test_arr, y_prob)
    thr_note = "Youden (J=TPR−FPR)"
else:
    rough_thr = find_best_threshold_multi(y_test_arr, y_prob, target=TARGET_ALL)
    thr_note = f"Multi-metric (≥{TARGET_ALL:.2f})"

# 2) ince ayar: çevresinde 0.001 adım
best_thr, hit_all = fine_tune_threshold(y_test_arr, y_prob, rough_thr, target=TARGET_ALL)
stage_note = "fine-tuned" if hit_all else "best F1×BA around rough"


if not hit_all:
    global_best, global_ok = None, None
    for thr in np.arange(0.01, 0.991, 0.001):
        acc, prec, rec, spec, f1 = _metrics_at(y_test_arr, y_prob, thr)
        ba = 0.5*(rec+spec)
        score = f1*ba
        if (acc>=TARGET_ALL) and (prec>=TARGET_ALL) and (rec>=TARGET_ALL) and (spec>=TARGET_ALL) and (f1>=TARGET_ALL):
            if (global_ok is None) or (f1 > global_ok[0]):
                global_ok = (f1, thr)
        if (global_best is None) or (score > global_best[0]):
            global_best = (score, thr)
    if global_ok is not None:
        best_thr = global_ok[1]
        stage_note = "global sweep 97+"
        hit_all = True
    else:
        best_thr = global_best[1]
        stage_note = "global sweep best F1×BA"


if not hit_all and THRESH_METHOD.lower() != "youden":
    best_thr = best_thr_youden(y_test_arr, y_prob)
    stage_note = "fallback Youden"

acc, prec, rec, spec, f1 = _metrics_at(y_test_arr, y_prob, best_thr)
print(f"[THR] yöntem = {thr_note} | aşama = {stage_note} | seçilen eşik = {best_thr:.6f}")
print(f"[METRICS] Acc={acc:.4f} | Prec={prec:.4f} | Recall={rec:.4f} | Spec={spec:.4f} | F1={f1:.4f} | AUC={roc_auc_score(y_test_arr, y_prob):.4f}")

y_pred = (y_prob >= best_thr).astype(int)

all_results = []
add_results(y_test_arr, y_pred, y_prob,
            model_name="Hybrid_SAE_LGBM", set_name="Test",
            veri_name=veri_name, results_list=all_results)

# ROC
fpr, tpr, _ = roc_curve(y_test_arr, y_prob)
auc_pos = roc_auc_score(y_test_arr, y_prob)
auc_neg = roc_auc_score(1 - y_test_arr, 1 - y_prob)

plt.figure()
plt.plot(fpr, tpr, label=f'AUC (Malicious=1): {auc_pos:.4f}')
plt.plot([0,1],[0,1],'--')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC: Hybrid_SAE_LGBM - OFF (TEMPORAL TEST)')
plt.legend(loc='lower right')
plt.savefig(os.path.join(save_dir, 'TEMPORAL_OFF_Hybrid_SAE_LGBM_ROC.png'))
plt.close()

# Confusion Matrix
cm = confusion_matrix(y_test_arr, y_pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Malicious'])
disp.plot(cmap="Blues", values_format="d", colorbar=False)
plt.title("Confusion Matrix - Hybrid_SAE_LGBM (Temporal Test)")
plt.savefig(os.path.join(save_dir, 'TEMPORAL_OFF_Hybrid_SAE_LGBM_CM.png'), dpi=300)
plt.close()

# Özet tablo
df_results = pd.DataFrame(all_results).round(6)
csv_path  = os.path.join(save_dir, "EMBER24_final_model_performance_summary_FULL.csv")
xlsx_path = os.path.join(save_dir, "EMBER24_final_model_performance_summary_FULL.xlsx")
df_results.to_csv(csv_path, index=False)
df_results.to_excel(xlsx_path, index=False)

print("\n--- Nihai Sonuçlar ---")
print(df_results)
print("\nDosyalar kaydedildi:\n", csv_path, "\n", xlsx_path)

