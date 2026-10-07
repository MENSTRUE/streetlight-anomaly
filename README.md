# 💡 StreetLight Anomaly

**4% — AI/ML Portfolio Progress**

StreetLight Anomaly adalah proyek *machine learning* untuk **mendeteksi dan memprioritaskan pola konsumsi energi yang tidak normal** agar kasus yang paling perlu diperiksa dapat masuk lebih dulu ke antrean teknisi.

Proyek ini **bukan sistem monitoring lampu jalan yang sudah terpasang di dunia nyata**. Dataset utama berasal dari **LEAD — Large-scale Energy Anomaly Detection**, yaitu benchmark anomali energi berbasis *building smart-meter*. Karena belum menggunakan telemetry lampu jalan nyata, aplikasi Streamlit pada repository ini berfungsi sebagai **simulasi operasional / prototype**.

> **Prinsip utama:** model membantu menentukan prioritas pemeriksaan.  
> Model **tidak menyatakan bahwa sebuah lampu pasti rusak**, dan skor anomali **bukan ukuran tingkat kerusakan fisik**.

---

## 1. Latar Belakang

Pada sistem penerangan jalan skala besar, operator atau teknisi berpotensi menangani banyak unit dan banyak pembacaan konsumsi energi setiap hari.

Jika seluruh pembacaan diperiksa secara manual, beberapa masalah dapat muncul:

- jumlah data terlalu besar untuk diperiksa satu per satu;
- pola abnormal dapat terlambat diketahui;
- teknisi tidak memiliki urutan kasus yang perlu diperiksa terlebih dahulu;
- pembacaan yang terlihat tidak biasa belum tentu benar-benar menunjukkan kerusakan.

Karena itu, proyek ini tidak mencoba menggantikan teknisi.

Tujuan sistem adalah:

```text
Data konsumsi energi
        ↓
Analisis pola temporal
        ↓
Skor / probabilitas anomali
        ↓
Prioritas pemeriksaan
        ↓
Technician Priority Queue
        ↓
Verifikasi manusia
```

Dengan pendekatan ini, *machine learning* digunakan sebagai **decision-support system**, bukan sebagai mesin keputusan final.

---

# 2. Tujuan Proyek

StreetLight Anomaly dikembangkan untuk mempelajari bagaimana sebuah sistem dapat:

1. mengenali pola konsumsi energi yang berbeda dari pola normal;
2. membandingkan beberapa pendekatan deteksi anomali;
3. mengevaluasi model menggunakan metode yang sesuai untuk data yang sangat tidak seimbang;
4. mengurutkan kasus berdasarkan prioritas pemeriksaan;
5. mensimulasikan bagaimana hasil model dapat digunakan oleh teknisi;
6. membangun fondasi yang nantinya dapat dikembangkan menggunakan telemetry lampu jalan nyata.

---

# 3. Tiga Aspek Utama yang Sudah Dikerjakan

Saat ini proyek memiliki **tiga bagian utama**: **data & feature engineering**, **model & evaluasi**, dan **aplikasi simulasi**.

---

## A. Data dan Causal Feature Engineering

### Dataset

Eksperimen menggunakan:

**LEAD — Large-scale Energy Anomaly Detection**

Dataset yang digunakan pada Kaggle:

```text
/kaggle/input/datasets/loliwibu/lead-large-scale-energy-anomaly-detection
```

File utama yang digunakan:

```text
train.csv
```

Kolom utama:

```text
building_id
timestamp
meter_reading
anomaly
```

Pada eksperimen ini terdapat sekitar:

- **1,64 juta pembacaan**
- **200 building**
- periode data sekitar **1 tahun**
- proporsi anomali sekitar **2%**

### Kenapa dataset ini dipakai?

Dataset lampu jalan publik dengan telemetry energi, label gangguan, dan skala besar cukup sulit diperoleh.

Karena itu LEAD digunakan sebagai **benchmark awal** untuk mempelajari:

- pola penggunaan energi;
- deteksi anomali pada time-series;
- class imbalance;
- generalisasi antar-unit;
- prioritas pemeriksaan.

Namun perlu ditegaskan:

> **building_id pada LEAD bukan ID lampu jalan.**

LEAD adalah dataset **building smart-meter**, sehingga hasil penelitian ini belum dapat dianggap sebagai performa nyata pada jaringan lampu jalan.

### Feature engineering

Semua fitur utama dibuat secara **causal**, yaitu hanya menggunakan data saat ini dan masa lalu.

Tidak ada *centered window* yang melihat masa depan.

Beberapa fitur yang digunakan:

#### Fitur waktu

```text
hour
day of week
month
weekend
hour_sin
hour_cos
dow_sin
dow_cos
```

Transformasi sinus/cosinus digunakan karena waktu memiliki sifat siklik.

Contoh:

```text
23:00 dekat dengan 00:00
Minggu dekat dengan Senin
```

---

#### Lag features

```text
lag_1
lag_2
lag_24
lag_168
```

Artinya model dapat membandingkan pembacaan saat ini dengan:

- 1 jam sebelumnya;
- 2 jam sebelumnya;
- 24 jam sebelumnya;
- 168 jam atau 1 minggu sebelumnya.

---

#### Rolling features

```text
rolling_mean_6
rolling_mean_24
rolling_std_24
rolling_min_24
rolling_max_24
```

Fitur ini membantu melihat apakah pembacaan sekarang berbeda dari perilaku beberapa jam sebelumnya.

---

#### Zero dan flatline behavior

```text
zero_count_24
flat_count_24
```

Fitur ini penting untuk mendeteksi pola seperti:

```text
sensor membaca 0 terlalu lama
nilai tidak berubah selama beberapa jam
```

---

#### Difference dan ratio

```text
diff_1
diff_24
diff_168

ratio_lag_1
ratio_lag_24
ratio_lag_168
```

Fitur ini menggambarkan seberapa besar perubahan dibandingkan histori sebelumnya.

---

#### Hour-of-week baseline

Model juga menggunakan pola normal berdasarkan kombinasi:

```text
building_id × hour_of_week
```

Statistik yang digunakan:

```text
median
MAD (Median Absolute Deviation)
robust_z
relative_to_baseline
```

Baseline dihitung **hanya dari bagian training**, sehingga data validation/test tidak digunakan untuk membuat referensi normal.

---

## B. Machine Learning dan Evaluasi

Eksperimen tidak langsung menggunakan model kompleks.

Model dikembangkan secara bertahap.

### Baseline 1 — Robust Z-Score

Pendekatan statistik digunakan sebagai baseline sederhana.

Tujuannya:

> Apakah deviasi dari pola normal saja sudah cukup untuk menemukan anomali?

---

### Baseline 2 — Isolation Forest

Isolation Forest digunakan sebagai pendekatan *unsupervised anomaly detection*.

Awalnya pendekatan ini merupakan model utama.

Eksperimen menunjukkan bahwa Isolation Forest memiliki sinyal yang nyata, tetapi performanya masih terbatas dibanding model supervised.

---

### Model utama — Gradient Boosting

Karena dataset LEAD menyediakan label:

```text
anomaly = 0 / 1
```

maka proyek kemudian menggunakan pendekatan supervised Gradient Boosting.

LightGBM digunakan jika tersedia pada environment.

Jika LightGBM tidak tersedia, notebook memiliki fallback:

```text
HistGradientBoostingClassifier
```

---

## Kenapa bukan LSTM?

LSTM Autoencoder sempat dieksplorasi.

Namun pendekatan tersebut tidak dipertahankan sebagai model utama karena:

- training lebih berat;
- lebih sulit distabilkan;
- membutuhkan pengelolaan sequence yang jauh lebih besar;
- hasil awal belum mengalahkan pendekatan tabular;
- Gradient Boosting memberikan trade-off performa, interpretabilitas, dan biaya komputasi yang lebih baik.

Pelajaran penting dari eksperimen ini:

> **Model yang lebih kompleks tidak otomatis menjadi model yang lebih baik.**

Pemilihan model harus berdasarkan bukti evaluasi.

---

# 4. Strategi Evaluasi

Dataset ini sangat tidak seimbang.

Proporsi anomali hanya sekitar:

```text
~2%
```

Karena itu **accuracy tidak digunakan sebagai metrik utama**.

Misalnya model yang selalu menjawab NORMAL dapat memperoleh accuracy tinggi, tetapi tidak memiliki kegunaan.

Metrik utama yang digunakan:

```text
PR-AUC
Precision
Recall
F1
Precision@K
Recall@K
Lift@K
Event Recall
False Alerts / Building / Week
```

ROC-AUC tetap dilaporkan, tetapi bukan metrik utama.

---

## Protocol 1 — Temporal Holdout

Tujuannya menjawab:

> Apakah model masih bekerja ketika digunakan pada waktu yang lebih baru?

Pembagian dilakukan secara kronologis dengan:

```text
train
↓
7-day purge gap
↓
validation
↓
7-day purge gap
↓
test
```

Dengan demikian, test berada di masa depan relatif terhadap training.

### Hasil temporal holdout

Base anomaly rate:

**2,02%**

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC | Event Recall |
|---|---:|---:|---:|---:|---:|---:|
| Robust-Z | 10,23% | 43,62% | 16,58% | 0,0831 | 0,7468 | 42,15% |
| Isolation Forest | 18,86% | 56,22% | 28,24% | 0,1447 | 0,8401 | 46,15% |
| Gradient Boosting | **51,95%** | **64,82%** | **57,67%** | **0,6693** | **0,9090** | 50,92% |
| Gradient Boosting + IF | 50,53% | 64,76% | 56,77% | 0,6675 | 0,9062 | **51,54%** |

Pada temporal holdout, **Gradient Boosting biasa sedikit lebih baik** untuk:

- Precision
- F1
- PR-AUC
- ROC-AUC
- beban false alert

Sementara Gradient Boosting + IF sedikit lebih tinggi pada Event Recall.

---

## Protocol 2 — Unseen-Building Holdout

Pengujian kedua memisahkan building berdasarkan ID.

Artinya building yang berada pada test **tidak pernah muncul dalam training**.

Tujuannya:

> Apakah model dapat bekerja pada unit baru yang belum pernah dilihat?

Base anomaly rate:

**2,20%**

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC | Event Recall |
|---|---:|---:|---:|---:|---:|---:|
| Robust-Z | 1,29% | 1,98% | 1,56% | 0,0324 | 0,6650 | 4,36% |
| Isolation Forest | 11,19% | 21,47% | 14,71% | 0,0810 | 0,7970 | 20,52% |
| Gradient Boosting | **90,41%** | **60,15%** | **72,24%** | **0,6276** | **0,8466** | **33,96%** |

False alert pada eksperimen ini sekitar:

**0,23 false-alert point per building per minggu.**

> Angka ini merupakan hasil benchmark historis pada LEAD, bukan klaim performa untuk lampu jalan nyata.

---

# 5. Capacity-Aware Evaluation

Dalam situasi operasional, teknisi mungkin hanya mampu memeriksa sebagian kecil kasus.

Karena itu proyek juga menggunakan:

```text
Precision@K
Recall@K
Lift@K
```

Contoh pertanyaan:

> Jika teknisi hanya mampu memeriksa 5% kasus dengan skor tertinggi, berapa banyak anomali yang dapat ditemukan?

Pendekatan ini lebih relevan dibanding sekadar bertanya:

```text
"Berapa accuracy model?"
```

karena sistem dirancang sebagai **priority-ranking system**.

---

# 6. Event-Level Evaluation

Anomali pada time-series sering muncul sebagai kejadian yang berlangsung beberapa jam.

Contoh:

```text
01:00 anomaly
02:00 anomaly
03:00 anomaly
04:00 anomaly
```

Empat baris tersebut dapat merupakan **satu kejadian**.

Karena itu proyek menghitung:

```text
Event Recall
```

Pertanyaan yang dijawab:

> Apakah sistem berhasil memberikan setidaknya satu alert pada sebuah kejadian anomali?

Hal ini membantu membuat evaluasi lebih dekat dengan pengalaman operator.

---

# 7. Aplikasi Simulasi Streamlit

Repository memiliki aplikasi:

```text
app.py
```

Aplikasi berfungsi sebagai **simulation dashboard**.

Aplikasi **tidak terkoneksi dengan sensor lampu jalan nyata**.

Dua sumber data tersedia:

```text
Generate demo
Upload CSV
```

---

## Generate Demo

Aplikasi membuat unit fiktif:

```text
SL-001
SL-002
SL-003
...
```

ID tersebut hanya digunakan untuk simulasi UI.

---

## Upload CSV

Format minimal:

```csv
unit_id,timestamp,meter_reading
SL-001,2026-10-01 00:00:00,72.4
SL-001,2026-10-01 01:00:00,71.8
```

atau:

```csv
building_id,timestamp,meter_reading
101,2026-10-01 00:00:00,72.4
101,2026-10-01 01:00:00,71.8
```

Disarankan menyediakan minimal:

```text
169 pembacaan per unit
```

karena model menggunakan:

```text
lag_168
```

---

# 8. Technician Priority Queue

Output utama aplikasi bukan sekadar:

```text
ANOMALY
NORMAL
```

Tetapi berupa ranking:

```text
priority_rank
unit_id
timestamp
meter_reading
anomaly_probability
status
reason
```

Contoh:

```text
1  SL-011  REVIEW
2  SL-002  REVIEW
3  SL-005  NORMAL
...
```

Tujuannya adalah membantu teknisi menentukan:

> **unit mana yang perlu diperiksa terlebih dahulu?**

---

# 9. Interpretasi Status

Status aplikasi terdiri dari:

```text
NORMAL
REVIEW
```

### NORMAL

Model belum menemukan bukti cukup kuat bahwa unit perlu diprioritaskan.

### REVIEW

Model menemukan pola yang cukup tidak biasa untuk masuk antrean pemeriksaan.

Namun:

```text
REVIEW ≠ confirmed fault
```

Artinya status REVIEW **bukan bukti bahwa perangkat rusak**.

Teknisi tetap perlu melakukan verifikasi.

---

# 10. Hal yang Tidak Diklaim oleh Proyek

StreetLight Anomaly tidak mengklaim:

- lampu jalan nyata sedang dipantau;
- adanya GPS realtime;
- adanya koneksi IoT aktif;
- adanya integrasi pemerintah;
- model dapat menentukan penyebab kerusakan fisik;
- anomaly probability sama dengan severity;
- hasil LEAD sama dengan performa pada jaringan lampu jalan sebenarnya.

Hal ini sengaja dijelaskan agar proyek tetap dapat dipertanggungjawabkan.

---

# 11. Struktur Repository

```text
streetlight-anomaly/
│
├── app.py
├── README.md
├── AGENTS.md
├── requirements.txt
├── sample_input.csv
│
├── models/
│   └── streetlight_anomaly_v6_bundle.joblib
│
├── notebooks/
│   └── StreetLight_Anomaly_v6_LightGBM_Benchmark.ipynb
│
├── reports/
│   ├── temporal_model_comparison.csv
│   ├── temporal_topk_comparison.csv
│   ├── grouped_model_comparison.csv
│   ├── grouped_topk_comparison.csv
│   ├── feature_importance.csv
│   └── final_summary.json
│
├── assets/
│   └── feature_importance.png
│
└── src/
    ├── __init__.py
    ├── features.py
    ├── predict.py
    └── simulation.py
```

---

# 12. Cara Menjalankan Project

Clone repository:

```bash
git clone https://github.com/MENSTRUE/streetlight-anomaly.git
cd streetlight-anomaly
```

Buat virtual environment:

```bash
python -m venv .venv
```

Aktifkan pada Windows:

```powershell
.venv\Scripts\activate
```

Install dependency:

```bash
pip install -r requirements.txt
```

Jalankan aplikasi:

```bash
streamlit run app.py
```

Kemudian buka:

```text
http://localhost:8501
```

---

# 13. Alur Pengembangan Project

Project ini tidak langsung menghasilkan model final.

Perjalanan eksperimennya kira-kira:

```text
Synthetic-data idea
        ↓
Public LEAD benchmark
        ↓
Robust statistical baseline
        ↓
Isolation Forest
        ↓
LSTM Autoencoder experiment
        ↓
Evaluation audit
        ↓
Causal feature engineering
        ↓
Gradient Boosting
        ↓
Temporal evaluation
        +
Unseen-building evaluation
        ↓
Operational simulation dashboard
```

Bagian ini penting karena menunjukkan bahwa proses ML bukan hanya:

```text
ambil dataset
train model
lihat accuracy
selesai
```

Tetapi:

```text
definisikan masalah
↓
pahami data
↓
buat baseline
↓
audit leakage
↓
pilih evaluasi
↓
bandingkan model
↓
analisis error
↓
buat prototype
```

---

# 14. Pelajaran yang Didapat

Beberapa pembelajaran utama dari proyek ini:

### 1. Accuracy tidak cukup untuk data imbalance

Dengan anomaly rate sekitar 2%, accuracy bisa terlihat tinggi meskipun model gagal mendeteksi anomaly.

Karena itu PR-AUC, Recall, Precision, dan Top-K lebih penting.

---

### 2. Evaluasi harus sesuai use case

Temporal holdout dan unseen-building holdout menjawab pertanyaan yang berbeda.

```text
Temporal:
bisakah model bekerja pada masa depan?

Grouped:
bisakah model bekerja pada unit baru?
```

---

### 3. Data leakage dapat membuat hasil menyesatkan

Baseline, scaler, Isolation Forest, dan statistik lainnya harus dihitung hanya dari training.

---

### 4. Model kompleks belum tentu lebih baik

LSTM lebih kompleks, tetapi Gradient Boosting lebih efektif untuk versi proyek sekarang.

---

### 5. Machine learning sebaiknya membantu manusia

Model tidak digunakan untuk menyatakan:

```text
"Lampu ini rusak."
```

tetapi:

```text
"Unit ini sebaiknya diperiksa lebih dahulu."
```

---

# 15. Roadmap — Agar Project Menjadi Lebih Sempurna

Versi sekarang adalah **MVP benchmark + operational simulation**.

Untuk berkembang menjadi sistem yang lebih realistis, roadmap berikut dapat dilakukan.

---

## Tahap 1 — Dataset Lampu Jalan Nyata

Prioritas terbesar adalah mendapatkan telemetry yang benar-benar berasal dari lampu jalan.

Idealnya setiap record memiliki:

```text
lamp_id
timestamp
voltage
current
power
energy
power_factor
temperature
on_off_state
maintenance_history
fault_label
```

Dengan dataset tersebut, `building_id` tidak lagi perlu digunakan sebagai proxy.

---

## Tahap 2 — Label Gangguan yang Lebih Detail

Saat ini target hanya:

```text
normal
anomaly
```

Ke depan dapat dikembangkan menjadi:

```text
lamp_off
power_spike
power_drop
flickering
sensor_failure
communication_failure
abnormal_consumption
normal
```

Dengan demikian sistem tidak hanya memberikan prioritas tetapi juga kandidat jenis gangguan.

---

## Tahap 3 — Forecasting Residual

Tambahkan model forecasting yang memprediksi konsumsi normal:

```text
historical readings
        ↓
forecast expected usage
        ↓
actual - expected
        ↓
forecast residual
```

Residual kemudian dapat digunakan sebagai fitur tambahan untuk classifier.

---

## Tahap 4 — Weather dan Calendar Context

Untuk penggunaan energi nyata, pola dapat dipengaruhi oleh:

```text
sunrise
sunset
weather
season
holiday
special events
```

Data tersebut dapat ditambahkan sebagai context feature.

---

## Tahap 5 — Probability Calibration

Probabilitas model dapat dikalibrasi menggunakan:

```text
Isotonic Regression
atau
Platt Scaling
```

Tujuannya agar nilai seperti:

```text
0.80
```

lebih konsisten secara probabilistik dan tidak hanya menjadi ranking score.

---

## Tahap 6 — Alert Post-Processing

Untuk mengurangi alert yang berkedip:

```text
NORMAL
REVIEW
NORMAL
REVIEW
NORMAL
```

dapat ditambahkan aturan:

```text
k-of-n consecutive observations
```

Contoh:

```text
3 dari 4 jam terakhir harus abnormal
```

sebelum alert dibuat.

Alert yang berdekatan juga dapat digabung menjadi satu event.

---

## Tahap 7 — Explainability

Tambahkan:

```text
SHAP
```

agar setiap alert dapat menjelaskan kontribusi fitur.

Contoh:

```text
REVIEW

Faktor terbesar:
+ konsumsi 4.1× lebih tinggi dari baseline
+ berbeda jauh dari waktu yang sama minggu lalu
+ rolling variance meningkat
```

Hal ini akan membuat sistem lebih mudah dipercaya oleh operator.

---

## Tahap 8 — Real IoT Integration

Arsitektur masa depan:

```text
Street-light smart meter
        ↓
MQTT / IoT gateway
        ↓
Message broker
        ↓
Feature service
        ↓
ML inference
        ↓
Alert database
        ↓
Technician dashboard
```

Model dapat dijalankan sebagai service:

```text
FastAPI
```

dan dashboard hanya menjadi frontend.

---

## Tahap 9 — Geospatial Monitoring

Jika tersedia lokasi asli, dashboard dapat ditambah:

```text
Map
↓
lamp location
↓
current status
↓
priority
↓
maintenance route
```

Namun fitur tersebut hanya boleh dibuat jika tersedia data lokasi sungguhan.

---

## Tahap 10 — Field Validation

Tahap paling penting sebelum sistem disebut production-ready:

```text
offline benchmark
↓
shadow deployment
↓
technician feedback
↓
false-alert analysis
↓
threshold adjustment
↓
limited pilot
↓
production evaluation
```

Performa pada dataset LEAD **tidak cukup** untuk membuktikan performa lapangan.

---

# 16. Target Arsitektur Masa Depan

```text
            ┌─────────────────────┐
            │ Street-Light Meter  │
            └──────────┬──────────┘
                       │
                       ▼
              ┌────────────────┐
              │ Telemetry API  │
              └───────┬────────┘
                      │
                      ▼
           ┌───────────────────────┐
           │ Time-Series Database  │
           └──────────┬────────────┘
                      │
                      ▼
            ┌────────────────────┐
            │ Feature Pipeline   │
            │ Causal / Realtime  │
            └─────────┬──────────┘
                      │
                      ▼
          ┌────────────────────────┐
          │ Anomaly Ranking Model  │
          └───────────┬────────────┘
                      │
                      ▼
          ┌────────────────────────┐
          │ Technician Priority    │
          │ Queue                  │
          └───────────┬────────────┘
                      │
                      ▼
             Human Verification
```

---

# 17. Status Project Saat Ini

| Komponen | Status |
|---|---|
| Public energy dataset | ✅ |
| Causal feature engineering | ✅ |
| Statistical baseline | ✅ |
| Isolation Forest baseline | ✅ |
| Gradient Boosting | ✅ |
| Temporal holdout | ✅ |
| Unseen-building holdout | ✅ |
| Top-K evaluation | ✅ |
| Event-level evaluation | ✅ |
| Streamlit simulation | ✅ |
| Upload CSV | ✅ |
| Technician priority queue | ✅ |
| Real street-light telemetry | ⏳ Future |
| Real IoT integration | ⏳ Future |
| Weather integration | ⏳ Future |
| SHAP explainability | ⏳ Future |
| Production deployment | ⏳ Future |
| Field validation | ⏳ Future |

---

# 18. Kesimpulan

StreetLight Anomaly saat ini bukan sebuah sistem smart-city yang sudah selesai.

Project ini adalah **ML benchmark + operational prototype** yang mempelajari bagaimana pembacaan energi dapat diubah menjadi prioritas inspeksi.

Tiga bagian utama yang sudah berhasil dibangun adalah:

```text
1. Data & causal feature engineering
2. Machine-learning benchmark & trustworthy evaluation
3. Streamlit operational simulation
```

Eksperimen menunjukkan bahwa pendekatan supervised Gradient Boosting jauh lebih efektif dibanding baseline statistik dan Isolation Forest pada LEAD.

Namun hasil tersebut tetap harus dibaca dalam konteks yang benar:

> **LEAD adalah building smart-meter benchmark, bukan telemetry lampu jalan nyata.**

Tahap selanjutnya adalah membawa metodologi ini ke data lampu jalan sungguhan, menambah context feature, memperbaiki explainability, dan melakukan validasi bersama teknisi.

---

## Catatan

**Model menghasilkan prioritas. Manusia melakukan verifikasi.**

---

## Author

**Wafa Bila Syaefurokhman**

GitHub: [MENSTRUE](https://github.com/MENSTRUE)
