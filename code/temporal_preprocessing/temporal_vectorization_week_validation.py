import json, pandas as pd

JSONL =r"C:\Users\Gaming\Desktop\temporal200k_subset.jsonl"
Y_PARQ = r"C:\Users\Gaming\Desktop\EMBER24_2000k.temporal_Vec_Clean\parquet\y_train_clean.parquet"

CHUNK = 100_000  # etiketler parça parça kontrol edildi - ram tasarrufu için

# Parquet etiketlerini oku
y = pd.read_parquet(Y_PARQ).iloc[:, 0].astype(int)

ok = 0
with open(JSONL, "r", encoding="utf-8") as f:
    buf = []
    for i, line in enumerate(f, 1):
        buf.append(int(json.loads(line)["label"]))
        if i % CHUNK == 0:
            j = i - CHUNK
            ok += (y[j:i].to_numpy() == buf).sum()
            buf = []
    #Son kalanları da kontrol et
    if buf:
        j = len(y) - len(buf)
        ok += (y[j:].to_numpy() == buf).sum()

print(f"Eşleşen toplam: {ok}/{len(y)}")


