# EMBER-Hybrid-Detection-SAE-TopK-LGBM

Bu depo, **EMBER 2024** veri seti üzerinde geliştirilen **SAE–Top-K–LightGBM** tabanlı hibrit bir mimari ile 
Windows PE dosyaları için **yüksek doğruluklu, düşük gecikmeli ve hafif** bir malware tespit sistemi 
oluşturmayı amaçlayan kapsamlı bir çalışmanın tüm kodlarını, deneysel analizlerini ve görsel çıktıları 
içermektedir.

Çalışma kapsamında; makine öğrenmesi, derin öğrenme ve hibrit yaklaşımlar karşılaştırmalı olarak ele alınmış, 
modeller **5-Fold çapraz doğrulama** ve **zaman bazlı (temporal) test** senaryoları altında değerlendirilmiştir. 
Elde edilen sonuçlar doğruluk, gecikme süresi, model boyutu ve çıkarım verimliliği gibi ölçütler üzerinden 
ayrıntılı biçimde raporlanmıştır.


##  1. Çalışmanın Amacı

Bu çalışmanın temel amacı, gerçek dünya koşullarında karşılaşılan büyük ölçekli zararlı yazılım 
verileri üzerinde, **hesaplama maliyeti düşük**, **genellenebilir** ve **gerçek zamanlı kullanıma uygun** 
bir malware tespit yaklaşımı geliştirmektir. Bu kapsamda, geleneksel makine öğrenmesi ve derin öğrenme 
yaklaşımlarının güçlü yönlerini bir araya getiren hibrit bir modelleme stratejisi benimsenmiştir.

Ayrıca, farklı değerlendirme senaryoları altında modellerin davranışlarını inceleyerek, 
zaman temelli veri kaymalarına karşı dayanıklılık ve pratik uygulanabilirlik açısından 
en uygun yaklaşımın belirlenmesi hedeflenmiştir.

Tüm deneyler iki ana senaryoda gerçekleştirilmiştir:

- **5-Fold Stratified Cross-Validation**
- **Temporal  Değerlendirme**

### Deneysel Kurguya İlişkin Not

Temporal (zaman bazlı) değerlendirme senaryosunda, yalnızca **OFF veri temsili** kullanılmıştır. 
Bu senaryoda, hesaplama maliyeti ve gerçekçi dağılım kayması koşulları dikkate alınarak, 
5-Fold çapraz doğrulama deneylerinde en yüksek ve en tutarlı performansı gösteren 
**iki ML**, **iki DL** ve **önerilen hibrit model** 
seçilmiş ve değerlendirilmiştir.

Bu tercih, temporal senaryonun gerçek dünya kullanım koşullarını daha doğru yansıtmasını 
hedefleyen deneysel tasarım kararının bir parçasıdır.



---

##  2. Klasör Yapısı
```
code/
├── 5Fold/
│   ├── DL_OFF_5Fold.py
│   ├── DL_PL5DA_5Fold.py
│   ├── DL_VAE_Robust_5Fold.py
│   ├── Hybrid_OFF_5Fold.py
│   ├── Hybrid_OFF_LGBM_5Fold.py
│   ├── Hybrid_OFF_SAE_LGBM_5Fold.py
│   ├── Hybrid_OFF_TopK_LGBM_5Fold.py
│   ├── Hybrid_PL5DA_5Fold.py
│   ├── Hybrid_VAE_5Fold.py
│   ├── ML_OFF_5Fold.py
│   ├── ML_PL5DA_5Fold.py
│   └── ML_VAE_5Fold.py
│
├── Temporal/
│   ├── DL_Temporal.py
│   ├── Hybrid_Temporal.py
│   └── ML_Temporal.py


results/
├── 5Fold/
│   ├── DL_…_summary.xlsx
│   ├── ML_…_summary.xlsx
│   ├── Hybrid_…_summary.xlsx
│   └── Hybrid_OFF_5FoldCV_bootstrap_CI_results.csv
│
├── Temporal/
│   ├── DL_Temporal_summary.xlsx
│   ├── ML_Temporal_summary.xlsx
│   └── Hybrid_Temporal_summary.xlsx


figures/
├── DL_5Fold/
├── ML_5Fold/
├── Hybrid_5Fold/
├── DL_Temporal/
├── ML_Temporal/
└── Hybrid_Temporal/

README.md 


```
---

##  3. Kullanılan Veri Seti

Bu çalışmada kullanılan veri seti **EMBER 2024** olup tamamen kamuya açıktır.

🔗 Veri seti bağlantısı:  
[https://github.com/elastic/ember](https://github.com/FutureComputing4AI/EMBER2024)

400.397 örnekten oluşan **dengeli alt küme**, araştırmacı tarafından oluşturulmuştur ve telif nedeniyle GitHub’da paylaşılmamaktadır.  
Veri hazırlama ve ön işleme adımları bu depo kapsamında açıklanmıştır.

---

##  4. Modeller

###  Makine Öğrenmesi (ML)
- CatBoost  
- Extra Trees  
- Gradient Boosting  
- HistGradientBoosting  
- SGDClassifier  

###  Derin Öğrenme (DL)
- Wide&Deep
- DNN  
- MLP_Mixer 
- ResnetMLP 
- gMLP-Tabular

###  Boyut İndirgeme Yöntemleri
- PLS-DA (Partial Least Squares Discriminant Analysis)
- VAE (Variational Autoencoder) 

###  Önerilen Hibrit Model
- **SAE (Autoencoder) – 512-256-256-Latent 256**
- **Top-K Feature Selection (LightGBM Importance)**
- **LightGBM Sınıflandırıcı**

---

##  5. Sonuçların Özeti

###  5-Fold CV Sonuçları (Hibrit)
- **Accuracy:** %98.30  
- **Recall:** %98.07  
- **Precision:** %98.53  
- **Specificity:** %98.54  
- **AUC-ROC:** %99.83  

### Temporal Test Sonuçları (Hibrit)
-  **Accuracy** 0.9670 
-  **Recall**   0.9610 
-  **Specificity**  0.9730 
-  **Precision**  0.9726 
-  **F1-Score**  0.9688 
-  **F1-Weighted** 0.9670
-  **AUC-ROC**  0.9933 

- **Latency:** 0.008 ms/örnek  
- **Throughput:** 120.180 örnek/s  
- **Model Boyutu:** 21.30 MB  
- **Eğitilebilir Parametre:** **3.03M**

---

## 6. Çalıştırma

Her Python dosyası bağımsız olarak terminal üzerinden çalıştırılabilir.

### Örnek kullanım:

python Hybrid_OFF_SAE_LGBM_5Fold.py

 Not: Model Adı Hakkında Açıklama

Bu projedeki bazı Excel sonuç dosyalarında "Model" kolonu otomatik olarak Hybrid_SAE_LGBM şeklinde görünmektedir.
Bu durum bazı senaryolarda yalnızca çıktı dosyasındaki etiketleme hatasından kaynaklanmakta olup, modelin gerçek yapısını etkilememektedir.

 OFF (5Fold) İçin Model Adı Doğrudur

OFF veri temsili (5Fold) için kullanılan nihai hibrit model gerçekten:
 Hybrid_SAE_LGBM
Dolayısıyla OFF sonuçlarında görülen model adı doğru ve geçerlidir.

 Ablasyon (Bileşen Analizi) Modelleri Hakkında

Bu çalışmada hibrit mimarinin bileşen katkılarını değerlendirmek amacıyla üç ayrı ablation modeli kullanılmıştır:

Hybrid_SAE_LGBM

Hybrid_TopK_LGBM

Hybrid_LGBM

Bazı ara çıktı dosyalarında bu üç model de Hybrid_SAE_LGBM olarak görünebilmektedir.
Bu, yalnızca etiketleme kaynaklı bir isimlendirme hatasıdır; performans metrikleri doğru model üzerinden hesaplanmıştır.

 Doğru Etiketlenmiş Excel Dosyaları Mevcuttur

Her bir ablation modeli için doğru şekilde etiketlenmiş Excel çıktı dosyaları ayrıca eklenmiştir.
Bu dosyalarda model adları doğru belirtilmiş olup, analiz ve karşılaştırmalar için güvenle kullanılabilir.

📧 İletişim
Sultan Tazefidan
📩 sultantazefidan.1@gmail.com

