import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.cross_decomposition import PLSRegression
from collections import defaultdict
import os
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
    y = pd.read_parquet(y_path)['label'].values  # etiket sütunu: 'label'

    # --- Teyit Kontrolleri ---
    y_sr = pd.read_parquet(y_path)['label']  # sadece kontrol (Series)
    print("Sınıf dağılımı:\n", y_sr.value_counts())
    assert set(y_sr.unique()) == {0, 1}, "Etiketler 0/1 değil!"
    # assert y_sr.value_counts().min() == y_sr.value_counts().max(), "Sınıflar dengeli değil!"

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
    clean_data = np.asarray(data)  # dtypeise 32 kalsın
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


def get_dl_predictions(model, X_data, threshold=0.5):
    y_prob = np.asarray(model.predict(X_data, verbose=0)).reshape(-1)
    y_pred = (y_prob >= threshold).astype(int)
    return y_pred, y_prob


# BÖLÜM 3: ANA DEĞERLENDİRME DÖNGÜSÜ

assert X.ndim == 2 and y.ndim == 1 and len(X) == len(y), "X/y boyutları uyuşmuyor."
cm_totals = defaultdict(lambda: np.zeros((2, 2), dtype=int))

print("\nBÖLÜM 3: 5-Katlı Çapraz Doğrulama ile Değerlendirme Başlatılıyor...")
all_results = []
cm_totals = defaultdict(lambda: np.zeros((2, 2), dtype=int))  # CM toplamları burada tanımlı olsun
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

y_test_cv_all, y_prob_cv_all, auc_folds = [], [], []

#  Eğitim süreleri için sözlük
import time
train_times = {}  # key: (veri_name, model_name, fold) -> value: seconds


# ---- DL model kurucular (parametreli) ----
def build_widedeep(inp, deep_units=(512, 256, 64), wide_units=1, act='relu'):
    x = Dense(deep_units[0], activation=act)(inp)
    x = Dense(deep_units[1], activation=act)(x)
    x = Dense(deep_units[2], activation=act)(x)
    wide = Dense(wide_units, activation='linear')(inp)
    merged = Concatenate()([x, wide])
    out = Dense(1, activation='sigmoid')(merged)
    return models.Model(inp, out, name="WideDeep")


def build_dnn(inp, units=(512, 256, 64), drop=0.2, act='relu'):
    x = Dense(units[0], activation=act)(inp);
    x = Dropout(drop)(x)
    x = Dense(units[1], activation=act)(x);
    x = Dropout(drop)(x)
    x = Dense(units[2], activation=act)(x)
    out = Dense(1, activation='sigmoid')(x)
    return models.Model(inp, out, name="DNN")


def build_mlp_mixer_tabular(input_dim, blocks=2, mix=128, hidden=256, dropout=0.1):
    inp = Input(shape=(input_dim,), name="x")
    x = inp
    for _ in range(blocks):
        # Feature-mixing
        h = layers.LayerNormalization()(x)
        h = layers.Dense(mix, activation='relu')(h)
        if dropout: h = layers.Dropout(dropout)(h)
        h = layers.Dense(input_dim)(h)
        x = layers.Add()([x, h])

        # Channel-mixing
        h = layers.LayerNormalization()(x)
        h = layers.Dense(hidden, activation='relu')(h)
        if dropout: h = layers.Dropout(dropout)(h)
        h = layers.Dense(input_dim)(h)
        x = layers.Add()([x, h])

    x = layers.LayerNormalization()(x)
    x = layers.Dense(64, activation='relu')(x)
    if dropout: x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(inp, out, name="MLP_Mixer_Tabular")


def _gmlp_block(x, dim_ff=256, dropout=0.1):
    h = layers.LayerNormalization()(x)
    u = layers.Dense(dim_ff, activation='relu')(h)
    g = layers.Dense(dim_ff, activation='sigmoid')(h)
    h = layers.Multiply()([u, g])
    if dropout: h = layers.Dropout(dropout)(h)
    h = layers.Dense(x.shape[-1])(h)
    return layers.Add()([x, h])


def build_gmlp_tabular(input_dim, blocks=2, dim_ff=256, dropout=0.1):
    inp = Input(shape=(input_dim,), name="x")
    x = inp
    for _ in range(blocks):
        x = _gmlp_block(x, dim_ff=dim_ff, dropout=dropout)

    x = layers.LayerNormalization()(x)
    x = layers.Dense(64, activation='relu')(x)
    if dropout: x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    return models.Model(inp, out, name="gMLP_Tabular_Tiny")


def build_resnet_mlp(inp, u1=512, u2=256, skip_dim=256, bottleneck=64, drop=0.3):
    h1 = Dense(u1, activation='relu')(inp);
    h1 = Dropout(drop)(h1)
    h2 = Dense(u2, activation='relu')(h1);
    h2 = Dropout(drop)(h2)
    skip = Dense(skip_dim)(h1)
    res = Add()([h2, skip]);
    res = Activation('relu')(res)
    z = Dense(bottleneck, activation='relu')(res)
    out = Dense(1, activation='sigmoid')(z)
    return models.Model(inp, out, name="ResNetMLP")


#  5-Katlı Çapraz Doğrulama
for fold, (train_index, test_index) in enumerate(skf.split(X, y), start=1):
    print(f"\n--- FOLD {fold}/5 ---")

    # Split
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # --- Ölçekleme  ---
    scaler = StandardScaler()
    X_train_scaled = bulletproof_clean(scaler.fit_transform(X_train))
    X_test_scaled = bulletproof_clean(scaler.transform(X_test))

    # --- Variance Threshold  ---
    vt = VarianceThreshold(0.0)
    X_train_scaled = vt.fit_transform(X_train_scaled)
    X_test_scaled = vt.transform(X_test_scaled)

    # --- Dataset haritası  ---
    pls = PLSRegression(
        n_components=64,
        scale=False,
        max_iter=1000,
        tol=1e-06
    )

    X_train_pls = bulletproof_clean(pls.fit_transform(X_train_scaled, y_train.astype(float))[0])
    X_test_pls = bulletproof_clean(pls.transform(X_test_scaled))
    print(f"PLS dtypes -> train: {X_train_pls.dtype}, test: {X_test_pls.dtype}")

    datasets = {
        "PLS-DA": (X_train_pls, y_train, X_test_pls),
    }
    #  Her veri temsili için DL modellerini eğit
    for veri_name, (X_train_curr, y_train_curr, X_test_curr) in datasets.items():
        print(f"Fold {fold} - Veri Tipi: {veri_name} - Modeller çalıştırılıyor...")

        # 1) Bulletproof temizlik
        X_train_final = bulletproof_clean(X_train_curr)
        X_test_final = bulletproof_clean(X_test_curr)

        # 2) NaN/Inf temizliği
        X_train_final = np.nan_to_num(X_train_final, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_final = np.nan_to_num(X_test_final, nan=0.0, posinf=0.0, neginf=0.0)

        # 3) DL girişleri (float32)
        dl_input_shape = (X_train_final.shape[1],)
        X_train_dl_base = X_train_final.astype(np.float32, copy=False)
        X_test_dl_base = X_test_final.astype(np.float32, copy=False)
        y_train_dl = y_train_curr.astype(np.float32, copy=False)
        # 4) Modeller
        inp = Input(shape=dl_input_shape)  # aynı Input bazı modeller için tekrar kullanılabilir
        dl_models = {
            "WideDeep": build_widedeep(inp, deep_units=(512, 256, 64)),
            "DNN": build_dnn(Input(shape=dl_input_shape), units=(512, 256, 64), drop=0.2),
            "MLP_Mixer": build_mlp_mixer_tabular(dl_input_shape[0], blocks=2, mix=128, hidden=256, dropout=0.1),
            "gMLP": build_gmlp_tabular(dl_input_shape[0], blocks=2, dim_ff=256, dropout=0.1),
            "ResNetMLP": build_resnet_mlp(Input(shape=dl_input_shape), u1=512, u2=256, bottleneck=64),
        }

        # 5) Eğitim/Degerleme döngüsü
        for name, model in dl_models.items():
            model.compile(optimizer='adam', loss='binary_crossentropy')

            # Girdi şekillendirme
            if name == "CNN1D":
                X_train_dl = np.expand_dims(X_train_dl_base, -1)  # (N, F, 1)
                X_test_dl = np.expand_dims(X_test_dl_base, -1)
            else:
                X_train_dl, X_test_dl = X_train_dl_base, X_test_dl_base

            # Eğitim süresini ölç
            t0 = time.time()
            model.fit(X_train_dl, y_train_dl, epochs=25, batch_size=256, verbose=0)
            train_sec = float(time.time() - t0)
            train_times[(veri_name, name, fold)] = train_sec
            print(f"[Süre] Veri:{veri_name} | Model:{name} | Fold:{fold} -> Eğitim süresi: {train_sec:.2f} sn")

            # Test seti
            y_pred_test, y_prob_test = get_dl_predictions(model, X_test_dl)
            add_results(y_test, y_pred_test, y_prob_test, name, "Test", veri_name, fold, all_results)
            print(f"Kaydedildi -> Veri:{veri_name}, Model:{name}, Set:Test")

            #  Toplama
            cm = confusion_matrix(y_test, y_pred_test, labels=[0, 1])
            cm_totals[("Test", veri_name, name)] += cm

            # Train seti
            y_pred_train, y_prob_train = get_dl_predictions(model, X_train_dl)
            add_results(y_train_curr, y_pred_train, y_prob_train, name, "Train", veri_name, fold, all_results)
            print(f"Kaydedildi -> Veri:{veri_name}, Model:{name}, Set:Train")

# BÖLÜM 4: Sonuçların Raporlanması (sadece kullanılan metriklerle)
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

    #  Eğitim süresini sonuç tablosuna ekle
    if {'Veri', 'Model', 'Fold'}.issubset(df_results.columns):
        df_results['Train_Time_Sec'] = df_results.apply(
            lambda r: train_times.get((r['Veri'], r['Model'], r['Fold'])), axis=1
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
            #  Eğitim süresi özetleri
            Train_Time_Sec_mean=('Train_Time_Sec', 'mean'),
            Train_Time_Sec_std=('Train_Time_Sec', 'std'),
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



