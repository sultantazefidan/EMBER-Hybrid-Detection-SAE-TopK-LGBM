import pandas as pd

X = pd.read_parquet(r"C:\Users\Gaming\Desktop\EMBER24_2000k.temporal_Vec_Clean\parquet\X_train.parquet")
y = pd.read_parquet(r"C:\Users\Gaming\Desktop\EMBER24_2000k.temporal_Vec_Clean\parquet\y_train.parquet")["label"]

#X ve y'yi birleştir
df = X.copy()
df["label"] = y.values

# Duplikatları kaldır
before = len(df)
df = df.drop_duplicates(keep="first")
after = len(df)
print("Silinen duplikat sayısı:", before - after)

# X ve y’yi ayır ve tekrar kaydet
y = df.pop("label")
X = df

X.to_parquet(r"C:\Users\Gaming\Desktop\EMBER24_2000k.temporal_Vec_Clean\parquet\X_train_clean.parquet", compression="zstd")
y.to_frame().to_parquet(r"C:\Users\Gaming\Desktop\EMBER24_2000k.temporal_Vec_Clean\parquet\y_train_clean.parquet", compression="zstd")

print("Temiz versiyon kaydedildi.")
