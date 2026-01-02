import tempfile
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
import numpy as np
import pandas as pd
from lightgbm import early_stopping
from sklearn.preprocessing import StandardScaler
import time
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from collections import defaultdict
import os
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_curve
from tensorflow.keras.layers import Embedding
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from tensorflow.keras import layers, models
from tensorflow.keras.layers import (
    Input, Dense, Dropout,
    Conv1D, MaxPooling1D,
    GlobalMaxPooling1D, GlobalAveragePooling1D,
    BatchNormalization,
    Add, Activation, Concatenate,
    LayerNormalization, MultiHeadAttention,
    Reshape,
)
import matplotlib;
matplotlib.use("Agg")
import matplotlib.pyplot as plt;
plt.ioff()


def bytes_to_mb(nbytes: int) -> float:
    return nbytes / (1024 ** 2)


def keras_model_size_mb(model: tf.keras.Model, dtype_bytes: int = 4) -> float:
    # sadece trainable ağırlıklar; float32 -> 4 byte
    n_params = int(np.sum([np.prod(w.shape) for w in model.trainable_weights]))
    return bytes_to_mb(n_params * dtype_bytes)


def lgbm_model_size_mb(clf) -> float:
    # LightGBM boyutunu geçici dosyaya kaydedip dosya boyutunu ölç
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
    tmp.close()
    clf.booster_.save_model(tmp.name)
    size_mb = bytes_to_mb(os.path.getsize(tmp.name))

    os.unlink(tmp.name)
    return size_mb


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


# BÖLÜM 3: ANA DEĞERLENDİRME DÖNGÜSÜ
assert X.ndim == 2 and y.ndim == 1 and len(X) == len(y), "X/y boyutları uyuşmuyor."
cm_totals = defaultdict(lambda: np.zeros((2, 2), dtype=int))

print("\nBÖLÜM 3: 5-Katlı Çapraz Doğrulama ile Değerlendirme Başlatılıyor...")
all_results = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_test_cv_all, y_prob_cv_all, auc_folds = [], [], []

for fold, (train_index, test_index) in enumerate(skf.split(X, y), start=1):
    print(f"\n--- FOLD {fold}/5 ---")

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Her adımdan sonra temizlik ile sağlamlaştırılmış boru hattı
    scaler = StandardScaler()
    X_train_scaled = bulletproof_clean(scaler.fit_transform(X_train))
    X_test_scaled = bulletproof_clean(scaler.transform(X_test))

    vt = VarianceThreshold(0.0)
    X_train_scaled = vt.fit_transform(X_train_scaled)
    X_test_scaled = vt.transform(X_test_scaled)

    datasets = {
        "OFF": (X_train_scaled, y_train, X_test_scaled),
    }

    for veri_name, (X_train_curr, y_train_curr, X_test_curr) in datasets.items():
        print(f"Fold {fold} - Veri Tipi: {veri_name} - Modeller çalıştırılıyor...")

        # 1) Bulletproof temizlik
        X_train_final = bulletproof_clean(X_train_curr)
        X_test_final = bulletproof_clean(X_test_curr)

        # 2) NaN/Inf temizliği
        X_train_final = np.nan_to_num(X_train_final, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_final = np.nan_to_num(X_test_final, nan=0.0, posinf=0.0, neginf=0.0)

        # HİBRİT MODEL (SAE + Top-K + LGBM)
        if veri_name == "OFF":
            print(
                f"[CHECK] OFF -> after scaler+VT: {X_train_scaled.shape}, "
                f"before AE (final32): {X_train_final.shape if 'X_train_final' in locals() else 'N/A'}"
            )

            # --- NF değişken isimleri ---
            input_dim_hybrid = X_train_final.shape[1]
            input_hybrid = Input(shape=(input_dim_hybrid,))

            # --- SAE (512 -> 256 -> 256 latent) + hafif denoising ---
            h1 = layers.Dense(512, activation='relu', name='sae_h1')(input_hybrid)
            h1 = layers.BatchNormalization()(h1)
            h1 = layers.Dropout(0.10)(h1)

            h2 = layers.Dense(256, activation='relu', name='sae_h2')(h1)
            h2 = layers.BatchNormalization()(h2)
            h2 = layers.GaussianNoise(0.01)(h2)

            encoded_hybrid = layers.Dense(256, activation='relu', name='sae_latent')(h2)

            # --- Decoder ---
            d1 = layers.Dense(256, activation='relu', name='sae_d1')(encoded_hybrid)
            d1 = layers.BatchNormalization()(d1)
            recon_out = layers.Dense(input_dim_hybrid, activation='linear', name='sae_recon')(d1)

            encoder_hybrid_model = models.Model(inputs=input_hybrid, outputs=encoded_hybrid, name="sae_encoder")
            autoencoder_hybrid = models.Model(inputs=input_hybrid, outputs=recon_out, name="sae_autoencoder")
            autoencoder_hybrid.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')

            # --- float32 ---
            X_train_final32 = X_train_final.astype(np.float32, copy=False)
            X_test_final32 = X_test_final.astype(np.float32, copy=False)

            print(
                f"[CHECK] OFF -> after scaler+VT: {X_train_scaled.shape}, dtype={X_train_scaled.dtype} | "
                f"before AE (final32): {X_train_final32.shape}, dtype={X_train_final32.dtype}"
            )

            # --- Zamanlayıcı (toplam eğitim) ---
            t0_train_total = time.perf_counter()

            # --- SAE eğit (erken durdurma) ---
            es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
            autoencoder_hybrid.fit(
                X_train_final32, X_train_final32,
                validation_data=(X_test_final32, X_test_final32),
                epochs=40, batch_size=512, verbose=0, callbacks=[es]
            )

            print(
                f"[CHECK] OFF -> after scaler+VT: {X_train_scaled.shape}, dtype={X_train_scaled.dtype} | "
                f"before AE (final32): {X_train_final32.shape}, dtype={X_train_final32.dtype}"
            )

            # --- Latent ---
            X_train_hybrid = encoder_hybrid_model.predict(X_train_final32, verbose=0)
            X_test_hybrid = encoder_hybrid_model.predict(X_test_final32, verbose=0)

            # --- Top-K (220) ---
            K_TOP = 220
            n_feat = X_train_final32.shape[1]
            K_eff = min(K_TOP, n_feat)

            probe_lgb = LGBMClassifier(
                objective="binary",
                n_estimators=1000,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
            )
            probe_lgb.fit(X_train_final32, y_train_curr)

            imp = np.asarray(getattr(probe_lgb, "feature_importances_", np.zeros(n_feat)), dtype=np.float64)
            if not np.any(imp):
                imp = np.arange(n_feat)

            top_idx = np.argsort(imp)[-K_eff:]

            Xtr_topK = X_train_final32[:, top_idx]
            Xte_topK = X_test_final32[:, top_idx]

            # latent + topK birleştir
            X_train_hybrid = np.hstack([X_train_hybrid, Xtr_topK]).astype(np.float32, copy=False)
            X_test_hybrid = np.hstack([X_test_hybrid, Xte_topK]).astype(np.float32, copy=False)

            # --- Classifier: LGBM ---
            lgbm_hybrid = LGBMClassifier(
                objective="binary",
                n_estimators=4000,
                learning_rate=0.02,
                num_leaves=72,
                max_depth=10,
                min_child_samples=140,
                min_split_gain=0.1,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                reg_lambda=5.0,
                reg_alpha=1.0,
                random_state=42,
                n_jobs=-1,
                colsample_bynode=0.8
            )

            lgbm_hybrid.fit(
                X_train_hybrid, y_train_curr,
                eval_set=[(X_test_hybrid, y_test)],
                eval_metric=["auc", "binary_logloss"],
                callbacks=[early_stopping(100, verbose=False)]
            )

            # BOYUT ÖLÇÜMÜ
            sae_mb = keras_model_size_mb(encoder_hybrid_model, dtype_bytes=4)  # float32
            lgbm_mb = lgbm_model_size_mb(lgbm_hybrid)
            total_mb = sae_mb + lgbm_mb
            print(f"[SIZE] SAE: {sae_mb:.2f} MB | LGBM: {lgbm_mb:.2f} MB | TOTAL: {total_mb:.2f} MB")

            # CİHAZ BİLGİSİ
            booster = lgbm_hybrid.booster_
            params = getattr(booster, "params", {})
            print("LGBM device:", params.get("device", params.get("device_type", "unknown")))

            # TEST İNFERANS ZAMANLAMA
            t0_inf = time.perf_counter()
            y_prob_test = lgbm_hybrid.predict_proba(
                X_test_hybrid, num_iteration=lgbm_hybrid.best_iteration_
            )[:, 1]
            infer_total = time.perf_counter() - t0_inf

            n_samples = X_test_hybrid.shape[0]
            latency_ms = (infer_total / max(n_samples, 1)) * 1000.0
            throughput = n_samples / infer_total if infer_total > 0 else float("inf")
            print(
                f"[INFER] total={infer_total:.3f} s | n={n_samples} | latency={latency_ms:.3f} ms/sample | throughput={throughput:.1f} samples/s")

            # Toplam eğitim süresi
            total_train_s = time.perf_counter() - t0_train_total
            print(f"[TIME] TOTAL TRAIN (SAE + LGBM): {total_train_s:.3f} s")

            booster = lgbm_hybrid.booster_
            params = getattr(booster, "params", {})
            print("LGBM device:", params.get("device", params.get("device_type", "unknown")))

            # --- TEST (eşik optimizasyonu: Youden) ---
            y_prob_test = lgbm_hybrid.predict_proba(
                X_test_hybrid, num_iteration=lgbm_hybrid.best_iteration_
            )[:, 1]


            def _best_thr(y_true, y_prob):
                fpr, tpr, thr = roc_curve(y_true, y_prob)
                j = tpr - fpr
                return thr[int(np.argmax(j))]


            thr = _best_thr(y_test, y_prob_test)

            y_pred_test = (y_prob_test >= thr).astype(int)
            add_results(
                y_test, y_pred_test, y_prob_test,
                model_name="Hybrid_SAE_LGBM", set_name="Test",
                veri_name=veri_name, fold=fold, results_list=all_results
            )
            auc_folds.append(roc_auc_score(y_test, y_prob_test))
            y_test_cv_all.append(y_test)
            y_prob_cv_all.append(y_prob_test)

            print(f"Kaydedildi -> Veri:{veri_name}, Model:Hybrid_SAE_LGBM, Set:Test")

            cm = confusion_matrix(y_test, y_pred_test, labels=[0, 1])
            cm_totals[("Test", veri_name, "Hybrid_SAE_LGBM")] += cm

            # --- TRAIN (aynı en iyi iterasyon ve eşik) ---
            y_prob_train = lgbm_hybrid.predict_proba(
                X_train_hybrid, num_iteration=lgbm_hybrid.best_iteration_
            )[:, 1]
            y_pred_train = (y_prob_train >= thr).astype(int)
            add_results(
                y_train_curr, y_pred_train, y_prob_train,
                model_name="Hybrid_SAE_LGBM", set_name="Train",
                veri_name=veri_name, fold=fold, results_list=all_results
            )
            print(f"Kaydedildi -> Veri:{veri_name}, Model:Hybrid_SAE_LGBM, Set:Train")

y_test_all = np.concatenate(y_test_cv_all)
y_prob_all = np.concatenate(y_prob_cv_all)

fpr, tpr, _ = roc_curve(y_test_all, y_prob_all)
auc_pos_total = roc_auc_score(y_test_all, y_prob_all)
auc_neg_total = roc_auc_score(1 - y_test_all, 1 - y_prob_all)

plt.figure()
plt.plot(fpr, tpr, label=f'AUC (Malicious=1): {auc_pos_total:.4f}')
plt.plot([0, 1], [0, 1], '--')
plt.xlabel('False Positive Rate');
plt.ylabel('True Positive Rate')
plt.title('ROC: Hybrid_SAE_LGBM - OFF (TOTAL CV)\n'
          f'AUC(Benign=0): {auc_neg_total:.4f}')
plt.legend(loc='lower right')
plt.savefig(os.path.join(save_dir, 'TOTAL_CV_OFF_Hybrid_SAE_LGBM_ROC.png'))
plt.close()

with open(os.path.join(save_dir, 'TOTAL_CV_OFF_Hybrid_SAE_LGBM_AUCs.txt'), 'w') as f:
    f.write(f"AUC (Malicious=1): {auc_pos_total:.6f}\n"
            f"AUC (Benign=0): {auc_neg_total:.6f}\n"
            f"Mean AUC over folds: {np.mean(auc_folds):.6f} "
            f"(±{np.std(auc_folds, ddof=0):.6f})\n")

# BÖLÜM 4: Tüm Değerlendirme Tamamlandı. Sonuçlar Raporlanıyor...
if 'all_results' in locals() and all_results:
    print("\n\nBÖLÜM 4: Tüm Değerlendirme Tamamlandı. Sonuçlar Raporlanıyor...")

    # TOPLAM CM'leri kaydet

    for (set_name, vname, mname), cm in cm_totals.items():
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Malicious'])
        disp.plot(cmap="Blues", values_format="d", colorbar=False)

        plt.title(f"Total Confusion Matrix\nModel: {mname} | Veri: {vname} | Set: {set_name}")
        fig = plt.gcf()
        fig.tight_layout()

        # Dosya kaydetme yolu (tek yerde tanımla)
        save_path = os.path.join(save_dir, f'TOTAL_{vname}_{mname}_CM.png')
        fig.savefig(save_path, dpi=300)

        plt.close(fig)
        print(f"Kaydedildi → {save_path}")

    # Sonuç tablosu
    df_results = (
        pd.DataFrame(all_results)
        .replace({np.inf: np.nan, -np.inf: np.nan})
        .dropna(subset=['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1 Score', 'F1 Weighted', 'AUC-ROC'])
    )

    # Kaç fold birleşmiş kontrol
    fold_counts = (
        df_results.groupby(['Veri', 'Model', 'Set'])['Accuracy']
        .count()
        .rename('n_folds')
    )

    # Özet: mean + std(tek gözlemde 0) + n_folds
    agg_std0 = lambda x: x.std(ddof=0)
    final_summary = (
        df_results
        .groupby(['Veri', 'Model', 'Set'], as_index=True)
        .agg(
            Accuracy_mean=('Accuracy', 'mean'), Accuracy_std=('Accuracy', agg_std0),
            Precision_mean=('Precision', 'mean'), Precision_std=('Precision', agg_std0),
            Recall_mean=('Recall', 'mean'), Recall_std=('Recall', agg_std0),
            Specificity_mean=('Specificity', 'mean'), Specificity_std=('Specificity', agg_std0),
            F1_mean=('F1 Score', 'mean'), F1_std=('F1 Score', agg_std0),
            F1W_mean=('F1 Weighted', 'mean'), F1W_std=('F1 Weighted', agg_std0),
            AUC_ROC_mean=('AUC-ROC', 'mean'), AUC_ROC_std=('AUC-ROC', agg_std0),
        )
        .join(fold_counts)
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

    # Uyarı: n_folds < 5 olan satırlar var mı?
    warn_rows = final_summary[final_summary['n_folds'] < 5]
    if not warn_rows.empty:
        print("\n[UYARI] Aşağıdaki gruplarda 5'ten az fold sonucu var; std düşük/0 olabilir:\n")
        print(warn_rows)
else:
    print("\n\nBÖLÜM 4: Raporlanacak sonuç bulunamadı.")



# BÖLÜM 5: BOOTSTRAP 95% GÜVEN ARALIĞI (CI) — Sessiz Mod (No Print)
import pandas as pd
def bootstrap_ci(y_true, y_prob, metric_fn, n_boot=1000, alpha=0.95):
    """Bootstrap güven aralığı hesaplar. Sessiz mod."""
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = np.random.randint(0, n, n)
        yt_bs = y_true[idx]
        yp_bs = y_prob[idx]
        y_pred_bs = (yp_bs >= 0.5).astype(int)
        stats.append(metric_fn(yt_bs, y_pred_bs))
    lower = np.percentile(stats, (1 - alpha) / 2 * 100)
    upper = np.percentile(stats, (1 + alpha) / 2 * 100)
    return np.mean(stats), lower, upper

# Tüm CV sonuçları (5-fold birleşik)
y_true_all = y_test_all
y_prob_all = y_prob_all

# Specificity fonksiyonu
def specificity_fn(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

# Hesaplanacak metrikler
metrics_ci = {
    "Accuracy": lambda yt, yp: accuracy_score(yt, yp),
    "Precision": lambda yt, yp: precision_score(yt, yp, zero_division=0),
    "Recall": lambda yt, yp: recall_score(yt, yp, zero_division=0),
    "Specificity": lambda yt, yp: specificity_fn(yt, yp),
    "F1-Score": lambda yt, yp: f1_score(yt, yp, zero_division=0),
    "F1-Weighted": lambda yt, yp: f1_score(yt, yp, average="weighted", zero_division=0),
}

ci_results = []

# Diğer tüm metrikler
for name, fn in metrics_ci.items():
    mean_ci, low, high = bootstrap_ci(
        y_true_all, y_prob_all, metric_fn=fn, n_boot=1000
    )
    ci_results.append({
        "Metric": name,
        "Mean": round(mean_ci, 6),
        "CI Lower (95%)": round(low, 6),
        "CI Upper (95%)": round(high, 6)
    })

# AUC-ROC (prob ile)
mean_auc, low_auc, high_auc = bootstrap_ci(
    y_true_all, y_prob_all,
    metric_fn=lambda yt, yp: roc_auc_score(yt, yp),
    n_boot=1000
)
ci_results.append({
    "Metric": "AUC-ROC",
    "Mean": round(mean_auc, 6),
    "CI Lower (95%)": round(low_auc, 6),
    "CI Upper (95%)": round(high_auc, 6)
})

# CSV olarak kaydet
ci_df = pd.DataFrame(ci_results)
csv_path = os.path.join(save_dir, "EMBER24_bootstrap_CI_results.csv")
ci_df.to_csv(csv_path, index=False)



