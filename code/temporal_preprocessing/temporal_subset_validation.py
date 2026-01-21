import json, os

SUBSET = r"C:\Users\Gaming\Desktop\temporal200k_subset.jsonl"
LOG    = r"C:\Users\Gaming\Desktop\temporal200k_subset_log.json"

def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

#1)Alan  kümesi ve sayıları
keys_union = set()
n_total = 0
n_label0 = 0
n_label1 = 0
sha_seen = set()
n_dup_sha = 0
n_dup_line = 0
lines_seen = set()

for obj in iter_jsonl(SUBSET):
    n_total += 1
    keys_union.update(obj.keys())

    #label sayımı (0/1 veya string "benign/malicious" ihtimaline karşı)
    v = obj.get("label")
    if isinstance(v, str):
        vv = 0 if v.lower().startswith("ben") else 1
    elif isinstance(v, bool):
        vv = 1 if v else 0
    else:
        try:
            vv = int(v)
        except Exception:
            vv = None

    if vv == 0: n_label0 += 1
    elif vv == 1: n_label1 += 1

    #duplikat kontrol (sha256 varsa)
    sha = obj.get("sha256") or obj.get("sha") or obj.get("hash")
    if sha is not None:
        if sha in sha_seen:
            n_dup_sha += 1
        else:
            sha_seen.add(sha)

    #satır bazında duplikat (tüm JSON aynı mı)
    s = json.dumps(obj, sort_keys=True)
    if s in lines_seen:
        n_dup_line += 1
    else:
        lines_seen.add(s)

print("1) JSON alanları (feature değil)")
print("Toplam satır:", n_total)
print("Alan (key) sayısı:", len(keys_union))

print("\n 2)Etiket dağılımı (subset) ")
print("label=0 (benign):   ", n_label0)
print("label=1 (malicious):", n_label1)

print("\n 3)Duplikat kontrolü")
print("Tekil sha256 sayısı:", len(sha_seen))
print("Tekrarlanan sha256 adedi:", n_dup_sha)
print("Satır bazında (tam JSON) duplikat:", n_dup_line)

#2)Hafta bazlı özet ve extra haftalar: LOG dosyasından
if os.path.exists(LOG):
    with open(LOG, "r", encoding="utf-8") as f:
        L = json.load(f)
    wk = L.get("week_summary", {})
    extras = L.get("extra_weeks_chosen", [])
    extra_info = L.get("extra_info", {})
    print("\n 2-b) Hafta bazında seçilen örnekler (LOG’tan) ")
    total_b = total_m = 0
    for week_name in sorted(wk.keys()):
        b = wk[week_name].get("ben_selected", 0)
        m = wk[week_name].get("mal_selected", 0)
        total_b += b; total_m += m
        print(f"{week_name}: benign={b} | malicious={m}")
    print("Toplam (LOG): benign=", total_b, "malicious=", total_m, "sum=", total_b+total_m)

    print("\nEkstra seçilen haftalar:", extras)
    for w in extras:
        info = extra_info.get(w, {})
        print(f"  {w}: +ben={info.get('ben_extra',0)}, +mal={info.get('mal_extra',0)}")
else:
    print("\nLOG bulunamadı; hafta bazlı sayımlar için log gerekir.")


print ("Haftalık kronoljık sıraya uygun mu? ")
import json
from datetime import datetime

LOG_FILE = r"C:\Users\Gaming\Desktop\temporal200k_subset_log.json"

with open(LOG_FILE, "r", encoding="utf-8") as f:
    log = json.load(f)

weeks = list(log["week_summary"].keys())

 #Haftalar tarih olarak sıralandı
def week_start(wname):
    return datetime.strptime(wname.split("_")[0], "%Y-%m-%d")

sorted_weeks = sorted(weeks, key=week_start)

 #Sıralama aynı mı kontrol et
if weeks == sorted_weeks:
    print("Haftalar kronolojik sıraya uygun (erken -> geç).")
else:
    print("Haftalar karışık sırada.")
    for i, (a, b) in enumerate(zip(weeks, sorted_weeks)):
        if a != b:
            print(f" - Fark: listedeki {a} yerine doğru sıra {b} olmalı.")
