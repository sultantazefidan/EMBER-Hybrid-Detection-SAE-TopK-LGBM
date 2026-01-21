import os, glob, json
import numpy as np
import pandas as pd

X_PATH = r"C:\Users\Gaming\Desktop\EMBER24_2000k.temporal_Vec_Clean\parquet\X_train.parquet"
Y_PATH = r"C:\Users\Gaming\Desktop\EMBER24_2000k.temporal_Vec_Clean\parquet\y_train.parquet"
OUT_DIR = r"C:\Users\Gaming\Desktop\EMBER24_2000k.temporal_Vec_Clean"

def safe_isfinite(df: pd.DataFrame) -> pd.Series:
    """NaN/Inf taraması - her ihtmiale karşı"""
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return pd.Series([0, 0], index=["nan", "inf"])
    nan_cnt = numeric.isna().sum().sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        inf_cnt = np.isinf(numeric.to_numpy()).sum()
    return pd.Series([int(nan_cnt), int(inf_cnt)], index=["nan", "inf"])

def check_feature_meta(dirpath: str):
    feats = sorted(glob.glob(os.path.join(dirpath, "feature_index*.json")))
    labels = sorted(glob.glob(os.path.join(dirpath, "label_map*.json")))
    print("\n7) Feature index / label map dosyaları")
    print("   feature_index*.json bulundu mu? ->", "Evet" if feats else "Hayır")
    print("   label_map*.json bulundu mu?     ->", "Evet" if labels else "Hayır")
    if feats:
        try:
            with open(feats[0], "r", encoding="utf-8") as f:
                fi = json.load(f)
            print(f" (Özet) feature_index ilk 3 kayıt: {list(fi.items())[:3]}")
            print(f" (Toplam feature sayısı - index'ten): {len(fi)}")
        except Exception as e:
            print("   feature_index okunamadı:", e)
    if labels:
        try:
            with open(labels[0], "r", encoding="utf-8") as f:
                lm = json.load(f)
            print(f" (Özet) label_map: {lm}")
        except Exception as e:
            print(" label_map okunamadı:", e)

def main():
    print("Dosyalar yükleniyor…")
    X = pd.read_parquet(X_PATH)
    y = pd.read_parquet(Y_PATH).iloc[:, 0]

    # 1) Şekil ve dağılım
    print("\n1) Şekil ve dağılım")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Özellik sayısı:", X.shape[1])
    print("Etiket dağılımı:", dict(y.value_counts().sort_index()))
    print("Toplam kayıt:", len(X))

    # Bellek kullanımı
    print("Toplam bellek (MB):", round(X.memory_usage(deep=True).sum() / 1e6, 2))

    # 2) NaN / Inf kontrolü
    print("\n2) NaN / Inf kontrolü")
    nan_inf = safe_isfinite(X)
    print(f"NaN sayısı: {nan_inf['nan']}")
    print(f"Inf sayısı: {nan_inf['inf']}")

    # 3) Duplikat kontrolü
    print("\n3) Duplikat kontrolü (tam satır bazında)")
    dup_rows = X.duplicated().sum()
    print("Duplikat satır sayısı:", int(dup_rows))

    # 4) SHA çakışması (vektörleştirilmişte genelde olmaz, atlanır)
    print("\n4) SHA çakışması kontrolü (vektörlerde atlandı)")
    print("SHA verisi vektör setinde bulunmadığı için geçildi.")

    # 5) Veri tipi kontrolü (float32?)
    print("\n5) X dtype kontrolü (hepsi float32 olmalı)")
    not_f32 = [c for c, dt in X.dtypes.items() if dt != np.float32]
    print("  float32 olmayan sütun sayısı:", len(not_f32))
    if not_f32:
        print("(İlk 10) float32 olmayan sütunlar:", not_f32[:10])

    # 6) Şekil / dtype güvenliği
    print("\n6) Şekil / dtype güvenliği")
    try:
        assert X.shape[0] == y.shape[0], "X ve y satır sayısı eşit değil!"
        y_safe = pd.to_numeric(y, errors="coerce").astype("Int8").fillna(-1).astype(np.int8)
        assert np.issubdtype(X.dtypes.values[0], np.number), "X sayısal değil!"
        print("Uzunluk ve tip kontrolleri geçti.")
    except AssertionError as e:
        print(" Hata:", e)

    # 7) Feature index / label map
    check_feature_meta(OUT_DIR)

    print("\n Kontroller Bitti")

if __name__ == "__main__":
    main()

