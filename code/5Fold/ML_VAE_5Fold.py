import os, time
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt; plt.ioff()
from scipy.special import expit
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.ensemble import (
    GradientBoostingClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier
)
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import IncrementalPCA

from catboost import CatBoostClassifier, CatBoostError
from catboost.utils import get_gpu_device_count

# TEMEL AYARLAR
# Temsil modu: "VAE", "AE-Light" veya "RPCA-64"
REP_MODE = "VAE"        #
USE_AE_LIGHT = True     #
AE_LIGHT_LATENT = 64

# CatBoost için cihaz tipi
_cat_task = "GPU" if get_gpu_device_count() > 0 else "CPU"

# MODELLER: AE-LIGHT
def build_ae_light(input_dim, latent_dim=64):
    inp = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(128, activation='relu')(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.05)(x)
    z = tf.keras.layers.Dense(latent_dim, activation='linear', name='z')(x)

    x2 = tf.keras.layers.Dense(128, activation='relu')(z)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    out = tf.keras.layers.Dense(input_dim, activation='linear')(x2)

    ae = tf.keras.Model(inp, out, name="ae_light")
    enc = tf.keras.Model(inp, z, name="ae_light_encoder")
    ae.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return ae, enc

# MODELLER: β-VAE
class _KLLossLayer(tf.keras.layers.Layer):
    """KL kaybını add_loss ile grafiğe ekleyen katman."""
    def __init__(self, beta_var, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta_var

    def call(self, inputs):
        mu, logvar = inputs
        # KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))  (örnek başına)
        kl = -0.5 * tf.reduce_sum(1.0 + logvar - tf.square(mu) - tf.exp(logvar), axis=-1)
        loss = self.beta * tf.reduce_mean(kl)
        self.add_loss(loss)
        # grafiğe bağlamak için dummy dön
        return tf.zeros_like(mu[:, :1])

class _BetaWarmup(tf.keras.callbacks.Callback):
    """β’yı ilk warmup_epochs boyunca 0→beta_final arasında lineer arttırır."""
    def __init__(self, warmup_epochs=10, beta_final=1.0, beta_var=None):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.beta_final = beta_final
        self.beta_var = beta_var

    def on_epoch_begin(self, epoch, logs=None):
        new_beta = min((epoch + 1) / self.warmup_epochs, 1.0) * self.beta_final
        self.beta_var.assign(tf.cast(new_beta, tf.float32))

def build_beta_vae(input_dim: int, latent_dim: int = 64, beta_final: float = 1.0, warmup_epochs: int = 10):
    """β-VAE: encoder (mu, logvar, sampling), decoder ve add_loss(KL).
       Dönüş: (vae_model, encoder_model, warmup_cb, es_cb)
    """
    inp = tf.keras.Input(shape=(input_dim,), name="vae_input")

    # Encoder
    x = tf.keras.layers.Dense(256, activation='relu', name='e_h1')(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.10)(x)

    x = tf.keras.layers.Dense(128, activation='relu', name='e_h2')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GaussianNoise(0.01)(x)

    mu = tf.keras.layers.Dense(latent_dim, name='mu')(x)
    logvar = tf.keras.layers.Dense(latent_dim, name='logvar')(x)

    # Reparameterization: z = mu + exp(0.5 * logvar) * eps
    def _sample(args):
        mu_, logvar_ = args
        eps = tf.random.normal(shape=tf.shape(mu_))
        return mu_ + tf.exp(0.5 * logvar_) * eps

    z = tf.keras.layers.Lambda(_sample, name="sampling")([mu, logvar])

    # Decoder
    d = tf.keras.layers.Dense(128, activation='relu', name='d_h1')(z)
    d = tf.keras.layers.BatchNormalization()(d)
    recon = tf.keras.layers.Dense(input_dim, activation='linear', name='recon')(d)

    # Modeller
    vae = tf.keras.Model(inp, recon, name="beta_vae")
    enc = tf.keras.Model(inp, mu, name="beta_vae_encoder")  # inference’ta mu kullanıldı

    # β değişkeni ve KL kaybı
    beta = tf.Variable(0.0, trainable=False, dtype=tf.float32, name='beta')
    _ = _KLLossLayer(beta, name="kl_loss")([mu, logvar])

    # Warm-up + ES
    warm = _BetaWarmup(warmup_epochs=warmup_epochs, beta_final=beta_final, beta_var=beta)
    es = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

    # Toplam kayıp: MSE (recon) + β·KL (add_loss ile ekleniyor)
    vae.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return vae, enc, warm, es

#O PATHS
save_dir = os.path.join(os.path.expanduser("~"), "Desktop", "EMBER RESULT")
os.makedirs(save_dir, exist_ok=True)

try:
    print("veri yukleniyor")

    X_path = r"C:\Users\Gaming\Desktop\EMBER24_400k.Main_Vec_Cleann\parquet\X_train_varcorr_clean.parquet"
    y_path = r"C:\Users\Gaming\Desktop\EMBER24_400k.Main_Vec_Cleann\parquet\y_train_clean.parquet"

    X = pd.read_parquet(X_path).values
    y = pd.read_parquet(y_path)['label'].values

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

# UTILS
def bulletproof_clean(data):
    finfo32 = np.finfo(np.float32)
    clean_data = np.asarray(data)
    clean_data = np.nan_to_num(clean_data, nan=0.0, posinf=finfo32.max, neginf=finfo32.min)
    return np.clip(clean_data, finfo32.min, finfo32.max)

def get_proba_bin(model, X):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba.ravel()
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return expit(scores)
    return model.predict(X).astype(float)

def add_results(y_true, y_pred, y_prob, model_name, set_name, veri_name, fold, results_list):
    y_prob_clean = bulletproof_clean(np.asarray(y_prob).ravel())
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    results_list.append({
        "Fold": fold, "Veri": veri_name, "Model": model_name, "Set": set_name,
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
        "n_estimators": 90, "learning_rate": 0.1, "max_depth": 6,
        "subsample": 0.8, "max_features": "sqrt", "random_state": 42,
    },
    "HistGB": { "max_iter": 100, "learning_rate": 0.1, "l2_regularization": 1.0 },
    "CatBoost": { "iterations": 300, "learning_rate": 0.1, "depth": 8 },
    "SGDClassifier": {
        "loss": "log_loss", "penalty": "elasticnet", "alpha": 1e-4, "l1_ratio": 0.15,
        "max_iter": 2000, "tol": 1e-3, "class_weight": "balanced", "early_stopping": True,
        "n_iter_no_change": 8, "learning_rate": "adaptive", "eta0": 0.01, "random_state": 42
    },
    "ExtraTrees": {
        "n_estimators": 500, "max_depth": None, "min_samples_split": 2,
        "min_samples_leaf": 1, "max_features": "sqrt", "bootstrap": False,
        "random_state": 42, "n_jobs": -1,
    },
}

# CV LOOP
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

    # Scale + VT
    scaler = RobustScaler()
    X_train_scaled = bulletproof_clean(scaler.fit_transform(X_train))
    X_test_scaled  = bulletproof_clean(scaler.transform(X_test))

    vt = VarianceThreshold(0.0)
    X_train_scaled = vt.fit_transform(X_train_scaled)
    X_test_scaled  = vt.transform(X_test_scaled)

    # Tek seçimli temsil dict'i
    datasets = { REP_MODE: (X_train_scaled, y_train, X_test_scaled) }

    for veri_name, (X_train_curr, y_train_curr, X_test_curr) in datasets.items():
        print(f"Fold {fold} - Veri Tipi: {veri_name} - Modeller çalıştırılıyor...")

        # Temizlik
        X_train_final = bulletproof_clean(X_train_curr)
        X_test_final  = bulletproof_clean(X_test_curr)
        X_train_final = np.nan_to_num(X_train_final, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_final  = np.nan_to_num(X_test_final,  nan=0.0, posinf=0.0, neginf=0.0)

        # TEMSİL HAZIRLAMA
        if veri_name == "AE-Light":
            X_train_final32 = X_train_final.astype(np.float32, copy=False)
            X_test_final32  = X_test_final.astype(np.float32,  copy=False)

            input_dim = X_train_final32.shape[1]
            ae_light, enc_light = build_ae_light(input_dim, latent_dim=AE_LIGHT_LATENT)

            es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            ae_light.fit(
                X_train_final32, X_train_final32,
                validation_data=(X_test_final32, X_test_final32),
                epochs=12, batch_size=512, verbose=0, callbacks=[es]
            )
            X_train_final = enc_light.predict(X_train_final32, batch_size=2048, verbose=0)
            X_test_final  = enc_light.predict(X_test_final32,  batch_size=2048, verbose=0)
            print(f"[AE-Light] -> latent shape: train {X_train_final.shape}, test {X_test_final.shape}")

        elif veri_name == "RPCA-64":
            rpca = IncrementalPCA(n_components=64, batch_size=4096)
            rpca.fit(X_train_final)
            X_train_final = rpca.transform(X_train_final)
            X_test_final  = rpca.transform(X_test_final)
            print(f"[RPCA-64] -> shape: train {X_train_final.shape}, test {X_test_final.shape}")

        elif veri_name == "VAE":
            X_train_final32 = X_train_final.astype(np.float32, copy=False)
            X_test_final32  = X_test_final.astype(np.float32,  copy=False)

            input_dim = X_train_final32.shape[1]
            latent_dim = 64
            vae, enc, warm, es = build_beta_vae(
                input_dim=input_dim, latent_dim=latent_dim,
                beta_final=1.0, warmup_epochs=10
            )
            vae.fit(
                X_train_final32, X_train_final32,
                validation_data=(X_test_final32, X_test_final32),
                epochs=25, batch_size=512, verbose=0,
                callbacks=[warm, es]
            )
            # Latent (mu)
            X_train_final = enc.predict(X_train_final32, batch_size=2048, verbose=0)
            X_test_final  = enc.predict(X_test_final32,  batch_size=2048, verbose=0)
            print(f"[VAE] -> latent shape: train {X_train_final.shape}, test {X_test_final.shape}")

        #ML MODELLERİ
        ml_models = {
            "GradientBoost": GradientBoostingClassifier(**best_params_store["GradientBoost"]),
            "HistGB": HistGradientBoostingClassifier(**best_params_store["HistGB"]),
            "CatBoost": CatBoostClassifier(
                **best_params_store["CatBoost"], task_type=_cat_task, devices="0",
                verbose=0, allow_writing_files=False
            ),
            "SGDClassifier": SGDClassifier(**best_params_store["SGDClassifier"]),
            "ExtraTrees": ExtraTreesClassifier(**best_params_store["ExtraTrees"]),
        }

        for name, model in ml_models.items():
            try:
                # TRAIN
                model.fit(X_train_final, y_train_curr)

                # TEST
                y_pred_test = model.predict(X_test_final)
                y_prob_test = get_proba_bin(model, X_test_final)
                add_results(y_test, y_pred_test, y_prob_test, name, "Test", veri_name, fold, all_results)
                cm = confusion_matrix(y_test, y_pred_test, labels=[0, 1])
                cm_totals[("Test", veri_name, name)] += cm

                # TRAIN METRİK
                y_pred_train = model.predict(X_train_final)
                y_prob_train = get_proba_bin(model, X_train_final)
                add_results(y_train_curr, y_pred_train, y_prob_train, name, "Train", veri_name, fold, all_results)

                print(f"Kaydedildi -> Veri:{veri_name}, Model:{name}")

            except CatBoostError as e:
                print(f"UYARI: {name} modeli '{veri_name}' verisiyle eğitilemedi. Hata: {e}. Bu model atlanıyor.")
            except Exception as e:
                print(f"BEKLENMEDİK HATA: {name} modeli '{veri_name}' verisiyle çalışırken çöktü. Hata: {e}")

# APORLAMA
if 'all_results' in locals() and all_results:
    print("\n\nBÖLÜM 4: Tüm Değerlendirme Tamamlandı. Sonuçlar Raporlanıyor...")

    os.makedirs(save_dir, exist_ok=True)

    print("\n--- Nihai (Toplam) Karışıklık Matrisleri Kaydediliyor ---")
    for (set_name, vname, mname), cm in cm_totals.items():
        try:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Malicious'])
            disp.plot(cmap="Blues", values_format="d", colorbar=False)
            plt.title(f"Total Confusion Matrix\nModel: {mname} | Veri: {vname} | Set: {set_name}")
            fig = plt.gcf(); fig.tight_layout()
            save_path = os.path.join(save_dir, f"TOTAL_{vname}_{mname}_CM.png")
            fig.savefig(save_path, dpi=300)
            plt.close(fig)
            print(f"Kaydedildi → {save_path}")
        except Exception as e:
            print(f"[UYARI] CM çiziminde hata ({vname}-{mname}): {e}")

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


