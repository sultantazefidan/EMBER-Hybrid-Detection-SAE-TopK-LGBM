import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
import os, time, random
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
    f1_score, confusion_matrix, ConfusionMatrixDisplay
)
def bytes_to_mb(nbytes: int) -> float:
    return nbytes / (1024**2)

def bulletproof_clean(data):
    finfo32 = np.finfo(np.float32)
    arr = np.asarray(data, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=finfo32.max, neginf=finfo32.min)
    return np.clip(arr, finfo32.min, finfo32.max)

def add_results(y_true, y_pred, y_prob, model_name, set_name, veri_name, fold, results_list):
    y_prob_clean = bulletproof_clean(np.asarray(y_prob).ravel())
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    results_list.append({
        "Veri": veri_name, "Model": model_name, "Set": set_name, "Fold": fold,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": specificity,
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "F1 Weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "AUC-ROC": roc_auc_score(y_true, y_prob_clean),
    })

def get_dl_predictions(model, X_data, threshold=0.5):
    y_prob = np.asarray(model.predict(X_data, verbose=0)).reshape(-1)
    y_pred = (y_prob >= threshold).astype(int)
    return y_pred, y_prob

# ----------------- Kayıt klasörü -----------------
save_dir = os.path.join(os.path.expanduser("~"), "Desktop", "EMBER RESULT")
os.makedirs(save_dir, exist_ok=True)

# ----------------- Veri Yolları -----------------
train_X_path = r"C:\Users\Gaming\Desktop\EMBER24_200k.Main_Vec_Cleann\parquet\X_train_varcorr_clean.parquet"
train_y_path = r"C:\Users\Gaming\Desktop\EMBER24_200k.Main_Vec_Cleann\parquet\y_train.parquet"
test_X_path  = r"C:\Users\Gaming\Desktop\EMBER24_200K.temporal_Vec_Clean\parquet\X_train_varcorr_clean.parquet"
test_y_path  = r"C:\Users\Gaming\Desktop\EMBER24_200K.temporal_Vec_Clean\parquet\y_test_clean.parquet"

OUT_X_ALIGNED = os.path.join(save_dir, "X_test_temporal_aligned.parquet")
OUT_LOG = os.path.join(save_dir, "feature_alignment_log.txt")

print(" Veri yükleniyor...")
X_train_df = pd.read_parquet(train_X_path).astype(np.float32)
y_train_df = pd.read_parquet(train_y_path)
X_test_df  = pd.read_parquet(test_X_path).astype(np.float32)
y_test_df  = pd.read_parquet(test_y_path)
print(f"X_train shape: {X_train_df.shape} | X_test (raw): {X_test_df.shape}")

# ----------------- Kolon hizalama -----------------
train_cols = X_train_df.columns.tolist()
test_cols  = X_test_df.columns.tolist()
missing_in_test = [c for c in train_cols if c not in test_cols]
extra_in_test   = [c for c in test_cols  if c not in train_cols]

for c in missing_in_test:
    X_test_df[c] = np.float32(0.0)
if extra_in_test:
    X_test_df = X_test_df.drop(columns=extra_in_test, errors="ignore")
X_test_df = X_test_df[train_cols]

assert X_test_df.shape[1] == X_train_df.shape[1], "Kolon sayısı eşleşmedi!"
assert list(X_test_df.columns) == train_cols, "Kolon sırası farklı!"
assert np.isfinite(X_test_df.to_numpy()).all(), "X_test içinde NaN/Inf var!"

pd.DataFrame(X_test_df, columns=train_cols).to_parquet(OUT_X_ALIGNED, index=False)
log_lines = [
    f"Train shape: {X_train_df.shape}",
    f"Test raw shape: {pd.read_parquet(test_X_path).shape}",
    f"Missing in test ({len(missing_in_test)}): {missing_in_test}",
    f"Extra in test   ({len(extra_in_test)}): {extra_in_test}",
    f"Aligned test shape: {X_test_df.shape}",
]
Path(OUT_LOG).write_text("\n".join(log_lines), encoding="utf-8")

# ----------------- y dizileri -----------------
y_train = (y_train_df['label'] if 'label' in y_train_df.columns else y_train_df.squeeze()).to_numpy().astype(int)
y_test  = (y_test_df['label']  if 'label'  in y_test_df.columns  else y_test_df.squeeze()).to_numpy().astype(int)

# ----------------- OFF temsili -----------------
scaler = StandardScaler()
X_train = bulletproof_clean(scaler.fit_transform(X_train_df))
X_test  = bulletproof_clean(scaler.transform(X_test_df))
vt = VarianceThreshold(0.0)
X_train = bulletproof_clean(vt.fit_transform(X_train))
X_test  = bulletproof_clean(vt.transform(X_test))
print(f"[CHECK OFF] after scaler+VT -> train: {X_train.shape}, test: {X_test.shape}")

#  DL Modelleri
input_dim = X_train.shape[1]

# ----  DNN ----
def build_dnn(input_dim, units=(512, 256, 128, 64), drop=0.3, act='relu'):
    i = layers.Input(shape=(input_dim,))
    x = layers.BatchNormalization()(i)
    x = layers.GaussianNoise(0.05)(x)

    for u in units:
        x = layers.Dense(u, activation=act,
                         kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(drop)(x)

    o = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(i, o, name="DNN")

# ---- gMLP  ----
def _gmlp_block(x, dim_ff=256, dropout=0.1):
    h = layers.LayerNormalization()(x)
    u = layers.Dense(dim_ff, activation='relu')(h)
    g = layers.Dense(dim_ff, activation='sigmoid')(h)
    h = layers.Multiply()([u, g])
    if dropout: h = layers.Dropout(dropout)(h)
    h = layers.Dense(x.shape[-1])(h)
    return layers.Add()([x, h])

def build_gmlp_tabular(input_dim, blocks=2, dim_ff=256, dropout=0.1):
    i = layers.Input(shape=(input_dim,))
    x = i
    for _ in range(blocks):
        x = _gmlp_block(x, dim_ff=dim_ff, dropout=dropout)
    x = layers.LayerNormalization()(x)
    x = layers.Dense(64, activation='relu')(x)
    if dropout: x = layers.Dropout(dropout)(x)
    o = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(i, o, name="gMLP")

dl_models = {
    "DNN":  build_dnn(input_dim),
    "gMLP": build_gmlp_tabular(input_dim),
}

#Eğitim/Test
VERI_NAME = "OFF"
EPOCHS = 25
BATCH = 256
all_results, cm_totals, train_times = [], {}, {}

Xtr, Xte = X_train.astype(np.float32), X_test.astype(np.float32)
ytr, yte = y_train.astype(np.float32), y_test.astype(np.float32)

# Seed ve class_weight
np.random.seed(42); tf.random.set_seed(42); random.seed(42)
pos, neg = float((ytr==1).mean()), 1.0 - float((ytr==1).mean())
CLASS_WEIGHTS = {0: 0.5/neg, 1: 0.5/pos}
cb_early = tf.keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=6, restore_best_weights=True)
cb_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5, patience=3, min_lr=1e-5)

for name, model in dl_models.items():
    if name == "DNN":
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='binary_crossentropy',
            metrics=[tf.keras.metrics.AUC(name='val_auc')]
        )
        t0 = time.time()
        model.fit(Xtr, ytr, epochs=60, batch_size=256, verbose=0,
                  validation_split=0.1, shuffle=True,
                  class_weight=CLASS_WEIGHTS,
                  callbacks=[cb_early, cb_lr])
        train_sec = time.time() - t0
    else:
        model.compile(optimizer='adam', loss='binary_crossentropy')
        t0 = time.time()
        model.fit(Xtr, ytr, epochs=EPOCHS, batch_size=BATCH, verbose=0)
        train_sec = time.time() - t0

    train_times[(VERI_NAME, name)] = train_sec
    print(f"[Süre] Veri:{VERI_NAME} | Model:{name} -> Eğitim süresi: {train_sec:.2f} sn")

    # Test
    y_pred_test, y_prob_test = get_dl_predictions(model, Xte)
    add_results(yte, y_pred_test, y_prob_test, name, "Test", VERI_NAME, 1, all_results)
    cm_totals[("Test", VERI_NAME, name)] = confusion_matrix(yte, y_pred_test, labels=[0,1])

    # Train
    y_pred_tr, y_prob_tr = get_dl_predictions(model, Xtr)
    add_results(ytr, y_pred_tr, y_prob_tr, name, "Train", VERI_NAME, 1, all_results)
    cm_totals[("Train", VERI_NAME, name)] = confusion_matrix(ytr, y_pred_tr, labels=[0,1])

#  Raporlama
if all_results:
    df_results = pd.DataFrame(all_results).replace([np.inf, -np.inf], np.nan).dropna(how='any')
    df_results['Train_Time_Sec'] = df_results.apply(lambda r: train_times.get((r['Veri'], r['Model'])), axis=1)

    # --- Karışıklık Matrisleri (Train & Test) kaydet ---
    print("\n--- Karışıklık Matrisleri Kaydediliyor ---")
    for (set_name, vname, mname), cm in cm_totals.items():
        try:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Malicious'])
            disp.plot(cmap="Blues", values_format="d", colorbar=False)
            plt.title(f"Confusion Matrix – {mname} | Veri: {vname} | Set: {set_name}")
            fig = plt.gcf();
            fig.tight_layout()
            out_path = os.path.join(save_dir, f"CM_{vname}_{mname}_{set_name}.png")
            fig.savefig(out_path, dpi=300)
            plt.close(fig)
            print(f"Kaydedildi → {out_path}")
        except Exception as e:
            print(f"[UYARI] CM çiziminde hata ({vname}-{mname}-{set_name}): {e}")

    summary = (
        df_results.groupby(['Veri','Model','Set'], as_index=True)
        .agg(
            Accuracy_mean=('Accuracy','mean'), Accuracy_std=('Accuracy','std'),
            Precision_mean=('Precision','mean'), Precision_std=('Precision','std'),
            Recall_mean=('Recall','mean'), Recall_std=('Recall','std'),
            Specificity_mean=('Specificity','mean'), Specificity_std=('Specificity','std'),
            F1_mean=('F1 Score','mean'), F1_std=('F1 Score','std'),
            F1W_mean=('F1 Weighted','mean'), F1W_std=('F1 Weighted','std'),
            AUC_ROC_mean=('AUC-ROC','mean'), AUC_ROC_std=('AUC-ROC','std'),
            Train_Time_Sec_mean=('Train_Time_Sec','mean')
        ).round(6)
    )

    csv_path = os.path.join(save_dir, "EMBER24_temporal_OFF_DNN_gMLP_summary.csv")
    xlsx_path = os.path.join(save_dir, "EMBER24_temporal_OFF_DNN_gMLP_summary.xlsx")
    summary.to_csv(csv_path)
    summary.to_excel(xlsx_path)
    print("\n--- Nihai Model Performans Özeti ---")
    print(summary)
    print("\nKaydedildi:\n", csv_path, "\n", xlsx_path)
else:
    print("! Raporlanacak sonuç bulunamadı.")


