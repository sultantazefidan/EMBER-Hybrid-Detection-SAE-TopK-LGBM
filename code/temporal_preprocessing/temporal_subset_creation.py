import os, glob, json, random, sys

BASE = r"C:\Users\Gaming\Desktop\ember2024_Data_Clean_Test"
OUT_DIR = r"C:\Users\Gaming\Desktop"
OUT_FILE = os.path.join(OUT_DIR, "temporal200k_subset.jsonl")
LOG_FILE = os.path.join(OUT_DIR, "temporal200k_subset_log.json")

PER_CLASS_PER_WEEK = 8333 # Her haftadan: 8333 benign + 8333 malicious
SEED = 42
INJECT_WEEK = True          # JSONL satırına "week" alanını entegre edildi

def detect_label_key(sample: dict):
    for k in ("label", "binary", "is_malicious", "malicious", "target", "y"):
        if k in sample:
            v = sample[k]
            if isinstance(v, (int, bool)) or (isinstance(v, str) and v in ("0", "1")):
                return k
    for k, v in sample.items():
        if isinstance(v, (int, bool)) or (isinstance(v, str) and v in ("0", "1")):
            return k
    return None

def to_int01(v):
    if isinstance(v, bool):
        return 1 if v else 0
    try:
        iv = int(v)
        return 1 if iv == 1 else 0
    except Exception:
        s = str(v).strip().lower()
        if s in ("1", "true", "malicious"):
            return 1
        return 0

def main():
    rnd = random.Random(SEED)
    files = sorted(glob.glob(os.path.join(BASE, "*_Win32_test.jsonl")))

    if not files:
        print("Hata: Test JSONL dosyası bulunamadı.")
        sys.exit(1)

    #Etiket anahtarları tespit edildi
    label_key = None
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    label_key = detect_label_key(json.loads(line))
                except Exception:
                    continue
                break
        if label_key:
            break
    if not label_key:
        print("Hata: Etiket (label) alanı tespit edilemedi.")
        sys.exit(1)
    print("Tespit edilen label anahtarı:", label_key)

    #Haftalık havuzlar
    week_records = {}
    week_order = []
    for fp in files:
        week_name = os.path.basename(fp)
        week_order.append(week_name)
        ben_lines, mal_lines = [], []
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if label_key not in obj:
                    continue
                v_int = to_int01(obj[label_key])
                (mal_lines if v_int == 1 else ben_lines).append(line)
        week_records[week_name] = {"ben": ben_lines, "mal": mal_lines}

    #Yeterlilik teyidi
    for wk, d in week_records.items():
        if len(d["ben"]) < PER_CLASS_PER_WEEK or len(d["mal"]) < PER_CLASS_PER_WEEK:
            print(f"Hata: {wk} içinde yeterli örnek yok "
                  f"(ben={len(d['ben'])}, mal={len(d['mal'])}, hedef={PER_CLASS_PER_WEEK}).")
            sys.exit(1)

    # Ana verisetinden subset oluşturma
    total_written = 0
    summary = {}

    with open(OUT_FILE, "w", encoding="utf-8") as out_f:
        for wk in week_order:
            d = week_records[wk]
            sel_ben = rnd.sample(d["ben"], PER_CLASS_PER_WEEK)
            sel_mal = rnd.sample(d["mal"], PER_CLASS_PER_WEEK)
            summary[wk] = {"ben_selected": len(sel_ben), "mal_selected": len(sel_mal)}

            for ln in sel_ben + sel_mal:
                if INJECT_WEEK:
                    obj = json.loads(ln)
                    obj["week"] = wk
                    out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                else:
                    out_f.write(ln)
                total_written += 1

    #Log yazma
    with open(LOG_FILE, "w", encoding="utf-8") as lf:
        json.dump({
            "base_dir": BASE,
            "out_file": OUT_FILE,
            "total_selected": total_written,
            "per_class_per_week": PER_CLASS_PER_WEEK,
            "week_summary": summary,
            "seed": SEED,
            "inject_week": INJECT_WEEK
        }, lf, indent=2, ensure_ascii=False)

    print(f" Subset JSONL yazıldı: {OUT_FILE}")
    print(f" Log yazıldı:          {LOG_FILE}")
    print(f" Toplam satır:         {total_written}")
    print(f" Not: Satırlar haftalık kronolojik sırayla yazıldı (shuffle yok).")

if __name__ == "__main__":
    main()
