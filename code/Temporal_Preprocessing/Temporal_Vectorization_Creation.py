import os, shutil, glob, inspect
from pathlib import Path
from multiprocessing import freeze_support
import numpy as np
import thrember

#Subset JSONL dosyası (main için oluşturduğum 200K satırlık dosya)
SUBSET_JSONL = Path(r"C:\Users\Gaming\Desktop\temporal200k_subset.jsonl")

#Çalışma ve çıktı klasörleri
WORK = Path(r"C:\Users\Gaming\Desktop\EMBER24_WOrk_usett")             #geçici alan
OUT  = Path(r"C:\Users\Gaming\Desktop\EMBER24_2000k.temporal_Vec_Clean")  #nihai vektörler
PARQ = OUT / "parquet"                                                  #parquet klasörü

# Parquet yazımındaki parça (chunk) boyutu
ROWS_PER_CHUNK = 50_000

def ensure_clean_dir(p: Path):
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)
    p.mkdir(parents=True, exist_ok=True)


def place_subset_and_placeholders(subset_jsonl: Path, work: Path):
    """
    - subset JSONL dosyasını, thrember'in beklediği 'train' adlandırmasına kopyalar
    - test/challenge için boş yer tutucu ekler
    """
    work.mkdir(parents=True, exist_ok=True)
    #train dosya adı (thrember create_vectorized_features train'i tanısın)
    train_name = "0000-00-00_0000-00-00_Win32_train.jsonl"
    shutil.copy2(subset_jsonl, work / train_name)

    #yer tutucular - geçici herhangi bir aksaklık olmasın diye çunku set bekliyor
    (work / "0000-00-00_0000-00-00_Win32_test.jsonl").write_text("", encoding="utf-8")
    (work / "Win32_challenge.jsonl").write_text("", encoding="utf-8")


def move_vectors(src: Path, dst: Path):
    """
    thrember'in ürettiği vektör ve eşlik eden mapping dosyalarını OUT'a taşır
    """
    dst.mkdir(parents=True, exist_ok=True)
    patterns = [
        "*.dat", "*.parquet", "*.npz", "*.npy",
        "*_index.json", "*feature_index*.json", "*label_map*.json"
    ]
    moved = 0
    for pat in patterns:
        for fp in glob.glob(str(src / pat)):
            if fp.lower().endswith(".jsonl"):
                continue
            shutil.move(fp, dst / os.path.basename(fp))
            moved += 1
    print(f"[MOVE] {moved} dosya OUT'a taşındı -> {dst}")


def read_train_vectors(vec_dir: Path):
    """
    thrember.read_vectorized_features ile TRAIN vektörlerini okur.
    Dönüş: X (float32), y (int8)
    """
    X_train, y_train = thrember.read_vectorized_features(str(vec_dir), subset="train")
    # RAM dostu tipler seçildi
    X_train = np.asarray(X_train, dtype=np.float32, order="C")
    y_train = np.asarray(y_train, dtype=np.int8,    order="C")
    print(f"[READ] X: {X_train.shape} (float32), y: {y_train.shape} (int8)")
    return X_train, y_train


def save_parquet_chunked(X: np.ndarray, y: np.ndarray, out_dir: Path, rows_per_chunk: int = 50_000):
    """
    X: np.ndarray [n, d] (float32), y: np.ndarray [n] (int8)
    ParquetWriter ile tek dosyaya güvenli, ardışık (chunk'lı) yazım. Sıkıştırma: ZSTD
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir.mkdir(parents=True, exist_ok=True)
    x_path = (out_dir / "X_train.parquet").as_posix()
    y_path = (out_dir / "y_train.parquet").as_posix()

    n, d = X.shape
    print(f"[PARQUET] X shape: {n} x {d} (float32, chunk={rows_per_chunk})")

    #X için writer (float32 şema)
    schema = pa.schema([pa.field(f"f{j}", pa.float32()) for j in range(d)])
    with pq.ParquetWriter(x_path, schema=schema, compression="zstd") as writer:
        for start in range(0, n, rows_per_chunk):
            end = min(start + rows_per_chunk, n)
            cols = [pa.array(X[start:end, j], type=pa.float32()) for j in range(d)]
            batch = pa.Table.from_arrays(cols, names=[f"f{j}" for j in range(d)])
            writer.write_table(batch)

    #Y için
    import pyarrow as pa
    import pyarrow.parquet as pq
    y_tbl = pa.Table.from_arrays([pa.array(y, type=pa.int8())], names=["label"])
    pq.write_table(y_tbl, y_path, compression="zstd")

    print(f"[PARQUET] yazıldı -> {x_path}\n[PARQUET] yazıldı -> {y_path}")


def main():
    if not SUBSET_JSONL.exists():
        raise SystemExit(f"Subset JSONL bulunamadı: {SUBSET_JSONL}")

    #1)WORK/OUT temiz kurulumu
    ensure_clean_dir(WORK)
    OUT.mkdir(parents=True, exist_ok=True)  #OUT'u silmeden oluştur

    #2)subset + placeholders
    place_subset_and_placeholders(SUBSET_JSONL, WORK)

    #3)thrember parametrelerini hazırlama
    sig = inspect.signature(thrember.create_vectorized_features)
    params = set(sig.parameters)
    kwargs = {}

    if "label_type" in params:
        kwargs["label_type"] = "label"  # ikili 0/1
    for k, v in (("subset", "train"), ("subsets", ["train"]), ("split", "train"), ("splits", ["train"])):
        if k in params:
            kwargs[k] = v
            break
    direct_out = False
    if "out_dir" in params:
        kwargs["out_dir"] = str(OUT)
        direct_out = True

    print("[RUN] thrember.create_vectorized_features kwargs ->", kwargs)
    thrember.create_vectorized_features(str(WORK), **kwargs)
    print(" Vektörleştirme tamamlandı.")

    #4)Kütüphane OUT'a yazmadıysa WORK’ten taşı
    if not direct_out:
        print("[INFO] out_dir desteklenmiyor -> WORK’ten OUT’a taşınıyor…")
        move_vectors(WORK, OUT)

    #5)TRAIN vektörlerini oku (numpy) ve Parquet’e güvenli yaz
    X, y = read_train_vectors(OUT)
    save_parquet_chunked(X, y, PARQ, rows_per_chunk=ROWS_PER_CHUNK)

    # 6)WORK’ü temizle
    shutil.rmtree(WORK, ignore_errors=True)
    print(" WORK temizlendi.")
    print(" Nihai klasörler:")
    print(" - Vektörler:", OUT)
    print(" - Parquet  :", PARQ)


if __name__ == "__main__":
    freeze_support()  # Windows multiprocessing için
    main()


