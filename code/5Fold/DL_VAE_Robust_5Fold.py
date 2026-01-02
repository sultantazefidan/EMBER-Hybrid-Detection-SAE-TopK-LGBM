import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
from tensorflow.keras import layers, models
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt; plt.ioff()
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt; plt.ioff()
import os, time
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt; plt.ioff()

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Add, Activation, Concatenate, LayerNormalization
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay
)
# Temsil modu: "VAE"
REP_MODE = "VAE"

# Kayıt klasörü
save_dir = os.path.join(os.path.expanduser("~"), "Desktop", "EMBER RESULT")
os.makedirs(save_dir, exist_ok=True)

             #1.bölümm
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


def bulletproof_clean(data):
    finfo32 = np.finfo(np.float32)
    clean_data = np.asarray(data)
    clean_data = np.nan_to_num(clean_data, nan=0.0, posinf=finfo32.max, neginf=finfo32.min)
    return np.clip(clean_data, finfo32.min, finfo32.max)

def get_proba_bin(model, X):
    # DL modelleri için
    y_prob = np.asarray(model.predict(X, verbose=0)).reshape(-1)
    y_pred = (y_prob >= 0.5).astype(int)
    return y_pred, y_prob

def add_results(y_true, y_pred, y_prob, model_name, set_name, veri_name, fold, results_list, train_time=None):
    y_prob_clean = bulletproof_clean(np.asarray(y_prob).ravel())
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    row = {
        "Fold": fold, "Veri": veri_name, "Model": model_name, "Set": set_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": specificity,
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "F1 Weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "AUC-ROC": roc_auc_score(y_true, y_prob_clean),
    }
    if train_time is not None:
        row["Train_Time_Sec"] = float(train_time)
    results_list.append(row)

# DL MODELLER
def build_widedeep(inp, deep_units=(512, 256, 64), wide_units=1, act='relu'):
    x = Dense(deep_units[0], activation=act)(inp)
    x = Dense(deep_units[1], activation=act)(x)
    x = Dense(deep_units[2], activation=act)(x)
    wide = Dense(wide_units, activation='linear')(inp)
    merged = Concatenate()([x, wide])
    out = Dense(1, activation='sigmoid')(merged)
    return models.Model(inp, out, name="WideDeep")

def build_dnn(inp, units=(512, 256, 64), drop=0.2, act='relu'):
    x = Dense(units[0], activation=act)(inp); x = Dropout(drop)(x)
    x = Dense(units[1], activation=act)(x);  x = Dropout(drop)(x)
    x = Dense(units[2], activation=act)(x)
    out = Dense(1, activation='sigmoid')(x)
    return models.Model(inp, out, name="DNN")

def build_mlp_mixer_tabular(input_dim, blocks=2, mix=128, hidden=256, dropout=0.1):
    inp = Input(shape=(input_dim,), name="x")
    x = inp
    for _ in range(blocks):
        # Feature-mixing
        h = LayerNormalization()(x)
        h = Dense(mix, activation='relu')(h)
        if dropout: h = Dropout(dropout)(h)
        h = Dense(input_dim)(h)
        x = Add()([x, h])
        # Channel-mixing
        h = LayerNormalization()(x)
        h = Dense(hidden, activation='relu')(h)
        if dropout: h = Dropout(dropout)(h)
        h = Dense(input_dim)(h)
        x = Add()([x, h])
    x = LayerNormalization()(x)
    x = Dense(64, activation='relu')(x)
    if dropout: x = Dropout(dropout)(x)
    out = Dense(1, activation='sigmoid')(x)
    return models.Model(inp, out, name="MLP_Mixer_Tabular")

def _gmlp_block(x, dim_ff=256, dropout=0.1):
    h = LayerNormalization()(x)
    u = Dense(dim_ff, activation='relu')(h)
    g = Dense(dim_ff, activation='sigmoid')(h)
    h = layers.Multiply()([u, g])
    if dropout: h = Dropout(dropout)(h)
    h = Dense(x.shape[-1])(h)
    return Add()([x, h])

def build_gmlp_tabular(input_dim, blocks=2, dim_ff=256, dropout=0.1):
    inp = Input(shape=(input_dim,), name="x")
    x = inp
    for _ in range(blocks):
        x = _gmlp_block(x, dim_ff=dim_ff, dropout=dropout)
    x = LayerNormalization()(x)
    x = Dense(64, activation='relu')(x)
    if dropout: x = Dropout(dropout)(x)
    out = Dense(1, activation='sigmoid')(x)
    return models.Model(inp, out, name="gMLP_Tabular_Tiny")

def build_resnet_mlp(inp, u1=512, u2=256, skip_dim=256, bottleneck=64, drop=0.3):
    h1 = Dense(u1, activation='relu')(inp); h1 = Dropout(drop)(h1)
    h2 = Dense(u2, activation='relu')(h1);  h2 = Dropout(drop)(h2)
    skip = Dense(skip_dim)(h1)
    res  = Add()([h2, skip]); res = Activation('relu')(res)
    z    = Dense(bottleneck, activation='relu')(res)
    out  = Dense(1, activation='sigmoid')(z)
    return models.Model(inp, out, name="ResNetMLP")

#β-VAE
class _KLLossLayer(tf.keras.layers.Layer):
    def __init__(self, beta_var, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta_var
    def call(self, inputs):
        mu, logvar = inputs
        kl = -0.5 * tf.reduce_sum(1.0 + logvar - tf.square(mu) - tf.exp(logvar), axis=-1)
        self.add_loss(self.beta * tf.reduce_mean(kl))
        return tf.zeros_like(mu[:, :1])

class _BetaWarmup(tf.keras.callbacks.Callback):
    def __init__(self, warmup_epochs=10, beta_final=1.0, beta_var=None):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.beta_final = beta_final
        self.beta_var = beta_var
    def on_epoch_begin(self, epoch, logs=None):
        new_beta = min((epoch + 1) / self.warmup_epochs, 1.0) * self.beta_final
        self.beta_var.assign(tf.cast(new_beta, tf.float32))

def build_beta_vae(input_dim: int, latent_dim: int = 64, beta_final: float = 1.0, warmup_epochs: int = 10):
    inp = tf.keras.Input(shape=(input_dim,), name="vae_input")

    # Encoder
    x = Dense(256, activation='relu', name='e_h1')(inp)
    x = layers.BatchNormalization()(x)
    x = Dropout(0.10)(x)
    x = Dense(128, activation='relu', name='e_h2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GaussianNoise(0.01)(x)

    mu = Dense(latent_dim, name='mu')(x)
    logvar = Dense(latent_dim, name='logvar')(x)

    # Reparameterization
    def _sample(args):
        mu_, logvar_ = args
        eps = tf.random.normal(shape=tf.shape(mu_))
        return mu_ + tf.exp(0.5 * logvar_) * eps
    z = layers.Lambda(_sample, name="sampling")([mu, logvar])

    # Decoder
    d = Dense(128, activation='relu', name='d_h1')(z)
    d = layers.BatchNormalization()(d)
    recon = Dense(input_dim, activation='linear', name='recon')(d)

    vae = models.Model(inp, recon, name="beta_vae")
    enc = models.Model(inp, mu,   name="beta_vae_encoder")

    beta = tf.Variable(0.0, trainable=False, dtype=tf.float32, name='beta')
    _ = _KLLossLayer(beta, name="kl_loss")([mu, logvar])

    warm = _BetaWarmup(warmup_epochs=warmup_epochs, beta_final=beta_final, beta_var=beta)
    es = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

    vae.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return vae, enc, warm, es

# BÖLÜM 3: 5-Katlı Çapraz Doğrulama ile Değerlendirme

#assert X.ndim == 2 and y.ndim == 1 and len(X) == len(y), "X/y boyutları uyuşmuyor."
cm_totals = defaultdict(lambda: np.zeros((2,2), dtype=int))

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

    # Tek seçimli temsil
    datasets = { REP_MODE: (X_train_scaled, y_train, X_test_scaled) }

    for veri_name, (X_train_curr, y_train_curr, X_test_curr) in datasets.items():
        print(f"Fold {fold} - Veri Tipi: {veri_name} - DL modeller çalıştırılıyor...")

        # Temizlik
        X_train_final = bulletproof_clean(X_train_curr)
        X_test_final  = bulletproof_clean(X_test_curr)
        X_train_final = np.nan_to_num(X_train_final, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_final  = np.nan_to_num(X_test_final,  nan=0.0, posinf=0.0, neginf=0.0)

        #TEMSİL
        if veri_name == "VAE":
            X_train32 = X_train_final.astype(np.float32, copy=False)
            X_test32  = X_test_final.astype(np.float32,  copy=False)
            input_dim = X_train32.shape[1]
            vae, enc, warm, es = build_beta_vae(
                input_dim=input_dim, latent_dim=64,
                beta_final=1.0, warmup_epochs=10
            )
            t0 = time.perf_counter()
            vae.fit(
                X_train32, X_train32,
                validation_data=(X_test32, X_test32),
                epochs=25, batch_size=512, verbose=0,
                callbacks=[warm, es]
            )
            vae_time = time.perf_counter() - t0
            # Latent (mu)
            X_train_final = enc.predict(X_train32, batch_size=2048, verbose=0)
            X_test_final  = enc.predict(X_test32,  batch_size=2048, verbose=0)
            print(f"[VAE] -> latent shape: train {X_train_final.shape}, test {X_test_final.shape} (train {vae_time:.2f}s)")

        elif veri_name == "RPCA-64":
            rpca = IncrementalPCA(n_components=64, batch_size=4096)
            t0 = time.perf_counter()
            rpca.fit(X_train_final)
            X_train_final = rpca.transform(X_train_final)
            X_test_final  = rpca.transform(X_test_final)
            rpca_time = time.perf_counter() - t0
            print(f"[RPCA-64] -> shape: train {X_train_final.shape}, test {X_test_final.shape} (train {rpca_time:.2f}s)")

        # DL girişleri
        dl_input_shape = (X_train_final.shape[1],)
        X_train_dl = X_train_final.astype(np.float32, copy=False)
        X_test_dl  = X_test_final.astype(np.float32,  copy=False)
        y_train_dl = y_train_curr.astype(np.float32, copy=False)

        # DL modelleri
        inp = Input(shape=dl_input_shape)
        dl_models = {
            "WideDeep":  build_widedeep(inp, deep_units=(512, 256, 64)),
            "DNN":       build_dnn(Input(shape=dl_input_shape), units=(512, 256, 64), drop=0.2),
            "MLP_Mixer": build_mlp_mixer_tabular(dl_input_shape[0], blocks=2, mix=128, hidden=256, dropout=0.1),
            "gMLP":      build_gmlp_tabular(dl_input_shape[0], blocks=2, dim_ff=256, dropout=0.1),
            "ResNetMLP": build_resnet_mlp(Input(shape=dl_input_shape), u1=512, u2=256, bottleneck=64),
        }

        #Eğitim / Değerlendirme
        for name, model in dl_models.items():
            model.compile(optimizer='adam', loss='binary_crossentropy')

            t0 = time.perf_counter()
            model.fit(X_train_dl, y_train_dl, epochs=25, batch_size=256, verbose=0)
            tr_time = time.perf_counter() - t0

            # TEST
            y_pred_test, y_prob_test = get_proba_bin(model, X_test_dl)
            add_results(y_test, y_pred_test, y_prob_test, name, "Test", veri_name, fold, all_results, train_time=tr_time)
            print(f"Kaydedildi -> Veri:{veri_name}, Model:{name}, Set:Test (train {tr_time:.2f}s)")

            cm = confusion_matrix(y_test, y_pred_test, labels=[0, 1])
            cm_totals[("Test", veri_name, name)] += cm

            # TRAIN
            y_pred_train, y_prob_train = get_proba_bin(model, X_train_dl)
            add_results(y_train_curr, y_pred_train, y_prob_train, name, "Train", veri_name, fold, all_results)

# RAPORLAMA
if 'all_results' in locals() and all_results:
    print("\n\nBÖLÜM 4: Tüm Değerlendirme Tamamlandı. Sonuçlar Raporlanıyor...")

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

    # --- Sonuç Tablosu ---
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
            Train_Time_Sec_mean=('Train_Time_Sec', 'mean'),
            Train_Time_Sec_std=('Train_Time_Sec', 'std'),
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


