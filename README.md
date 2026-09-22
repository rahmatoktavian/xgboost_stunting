# Prediksi Stunting dengan XGBoost

Proyek ini melatih model klasifikasi XGBoost untuk memprediksi status stunting berdasarkan data survei baduta. Script utama membaca Excel dan codebook, menyeleksi delapan fitur, melakukan validasi dan pengujian, lalu menyimpan model, metrik, dan grafik interpretasi.

## Struktur folder

```text
xgboost_stunting/
├── data/
│   ├── Data600.xlsx
│   └── Workbook.docx
├── script/
│   └── XGBoost.py
├── result/
│   ├── confusion_matrix_data600.png
│   ├── cross_validation_metrics_data600.png
│   ├── heatmap_correlation_data600_en.png
│   ├── heatmap_correlation_data600_id.png
│   ├── precision_recall_curve_data600.png
│   ├── roc_curve_data600.png
│   ├── shap_beeswarm_data600_en.png
│   └── shap_beeswarm_data600_id.png
└── README.md
```

| Lokasi | Fungsi |
| --- | --- |
| `data/Data600.xlsx` | Dataset sumber. Script membaca sheet pertama. File yang diperiksa berisi **222 baris dan 280 kolom**; angka `600` pada nama file bukan jumlah baris aktual. |
| `data/Workbook.docx` | Codebook untuk menampilkan nama variabel dan arti kode kategori pada keluaran. |
| `script/XGBoost.py` | Seluruh proses pemuatan data, pembersihan, seleksi fitur, pelatihan, evaluasi, interpretasi, dan penyimpanan artefak. |
| `result/` | Delapan grafik PNG yang **sudah tersedia** dalam folder saat dokumentasi ini dibuat. |

Script saat ini menulis keluaran baru ke folder `output_data600_fixed_xgboost/` yang ditentukan oleh variabel `OUTPUT_DIR`, bukan ke `result/`. Folder keluaran baru akan dibuat otomatis. Bagian [Menjalankan](#menjalankan) menjelaskan pengaturan path agar cocok dengan struktur repositori di atas.

## Alur metode sesuai script

1. **Memuat data dan codebook.** `pandas.read_excel` membaca sheet pertama `Data600.xlsx`; `python-docx` membaca tabel `Workbook.docx` untuk label variabel dan kategori. Nama kolom dibersihkan dan dibuat unik. Script mengharuskan kolom Excel `JT` sebagai target dengan nilai asli `0 = Stunting` dan `1 = Normal`. Untuk model, label dibalik menjadi `1 = Stunting` dan `0 = Normal`. Pool calon prediktor berasal dari kolom `A` sampai `JS`.
2. **Membagi data.** `train_test_split` membuat 80% data latih dan 20% data uji dengan stratifikasi target dan `random_state=42`. Dataset yang diperiksa memiliki 222 baris: pembagian tersebut menghasilkan 177 baris latih dan 45 baris uji. Data uji disimpan untuk evaluasi akhir.
3. **Membersihkan dan mengodekan fitur di dalam pipeline.** Fitur identitas, lokasi, dan beberapa turunan antropometri dibuang. Kolom `H` (panjang/tinggi anak) dan `JS` (HAZ) dikeluarkan untuk menghindari kebocoran target; `JQ` (WAZ) tetap boleh menjadi kandidat pada model utama. Pembersih juga membuang fitur yang seluruhnya kosong, lebih dari 90% kosong, atau konstan pada data latih. Nilai survei `77`, `88`, dan `888` diberi kategori khusus pada fitur nonkontinu. Pada fitur kontinu, hanya `888` dan `8888` yang dijadikan nilai hilang agar ukuran valid seperti `77` tetap utuh. Jawaban multi-pilihan dipisahkan menjadi indikator, fitur kategorikal diisi kategori `Missing` dan di-*one-hot*, sedangkan numerik yang hilang diteruskan ke XGBoost tanpa imputasi. Pembersihan dipelajari ulang di setiap proses pelatihan/fold.
4. **Memilih delapan fitur dan memvalidasi model.** Model awal memakai seluruh kandidat yang lolos pembersihan. Kontribusi SHAP absolut rata-rata dari fitur hasil transformasi dijumlahkan kembali ke fitur mentah, lalu delapan fitur teratas dipilih. Pada **nested stratified cross-validation**, proses seleksi dan pencarian parameter diulang pada setiap dari 5 *outer folds*. Setiap fold memakai `RandomizedSearchCV` dengan 20 kandidat parameter, 3 *inner folds*, dan `roc_auc` sebagai skor pencarian. `scale_pos_weight` dihitung dari data latih outer fold sebelum pencarian parameter; nilainya dipakai pada inner folds dalam pencarian tersebut. Jika `RUN_RISK_FACTOR_SCENARIO=True`, script juga menjalankan skenario pembanding tanpa `JQ` (WAZ) dan `G` (berat badan anak); skenario ini menghasilkan metrik CV tambahan, bukan model final terpisah.
5. **Melatih model akhir dan memilih ambang.** Delapan fitur dipilih ulang hanya dari split latih; pencarian parameter dijalankan ulang dan pipeline terbaik dilatih pada split tersebut. Probabilitas prediksi *out-of-fold* dari data latih digunakan untuk mencari ambang yang memaksimalkan *balanced accuracy* dan F1 pada rentang 0,10–0,90. Ambang ini tidak dipilih dari data uji.
6. **Menguji dan mengevaluasi.** Pada data uji, script menghitung metrik untuk ambang standar `0,5` dan ambang *balanced accuracy* hasil tuning. Metriknya mencakup akurasi, *balanced accuracy*, presisi, sensitivitas/recall stunting, spesifisitas, F1, ROC-AUC, dan *average precision* yang diberi label PR-AUC pada grafik. Confusion matrix yang digambar oleh **kode saat ini** memakai ambang *balanced accuracy* hasil tuning.
7. **Membuat EDA dan interpretasi.** Setelah evaluasi, script menyimpan distribusi target seluruh dataset. Korelasi dan heatmap dihitung dari fitur hasil transformasi **data latih**. ROC, precision–recall, confusion matrix, SHAP beeswarm, dan analisis kinerja subkelompok menggunakan **data uji**. Nilai SHAP menjelaskan kontribusi fitur terhadap keluaran model untuk kelas `Stunting` (`1`); nilai positif mendorong prediksi ke kelas tersebut. Korelasi dan SHAP tidak membuktikan hubungan sebab-akibat.
8. **Menyimpan hasil.** Script menulis CSV, JSON, PNG, serta model serialisasi `.pkl`. `metrics_data600.json` menyimpan rincian evaluasi dan konfigurasi. `model_metadata_data600.json` menyimpan metadata model dan lingkungan. `frontend_feature_schema_data600.json` serta *bundle* `.pkl` ditujukan untuk integrasi frontend; repository ini sendiri belum memuat aplikasi frontend.

> **Catatan jumlah data:** `FRONTEND_TOTAL_DATASET = 696` adalah konstanta tampilan di script. Nilai tersebut tidak berasal dari jumlah baris aktual `Data600.xlsx` yang diperiksa (222). Gunakan `training_rows`, `testing_rows`, atau `model_input_rows` pada keluaran untuk mengetahui jumlah data yang benar-benar dipakai.

## Instalasi

Gunakan Python **3.10 atau lebih baru** karena script memakai anotasi tipe `str | None`. Dari root proyek:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas openpyxl python-docx matplotlib seaborn shap scikit-learn xgboost joblib
```

Di Windows, aktifkan lingkungan virtual dengan `.venv\Scripts\activate` sebelum menjalankan perintah `pip`. Paket di atas mengikuti impor dalam script; repository yang diperiksa belum menyediakan `requirements.txt` atau versi dependensi yang dipatok. Untuk reproduksibilitas penelitian, catat versi paket dari lingkungan yang berhasil menjalankan analisis.

## Menjalankan

**Sesuaikan path terlebih dahulu.** Kode saat ini menganggap `Data600.xlsx` dan `Workbook.docx` berada di folder yang sama dengan script. Jika Excel tidak ditemukan, kode beralih ke path `//xgboost_stunting`, sehingga struktur folder repo di atas belum dapat dijalankan langsung. Ubah blok path di bagian awal `script/XGBoost.py` menjadi:

```python
PROJECT_DIR = Path(__file__).resolve().parent.parent / "data"
EXCEL_FILE = PROJECT_DIR / "Data600.xlsx"
CODEBOOK_FILE = PROJECT_DIR / "Workbook.docx"
OUTPUT_DIR = PROJECT_DIR.parent / "output_data600_fixed_xgboost"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
```

Setelah itu jalankan dari root proyek:

```bash
python script/XGBoost.py
```

Hasil baru akan muncul di `output_data600_fixed_xgboost/`. Script tidak menyediakan argumen CLI atau mode interaktif. Dua *mode* eksperimen diatur oleh konstanta pada awal file: model utama selalu memakai WAZ sebagai kandidat, sedangkan `RUN_RISK_FACTOR_SCENARIO` mengaktifkan atau menonaktifkan nested CV tambahan tanpa WAZ dan berat badan anak. Pelatihan membutuhkan waktu karena seleksi SHAP dan pencarian parameter dijalankan berulang pada fold CV.

Untuk memublikasikan repositori, tinjau izin penggunaan data terlebih dahulu: dataset berisi kolom identitas yang memang dikeluarkan dari model, tetapi masih tersimpan dalam file Excel mentah. Hindari mengunggah data mentah ke repositori publik tanpa izin yang sesuai.

## Makna file yang sudah ada di `result/`

| File | Cara membaca |
| --- | --- |
| `cross_validation_metrics_data600.png` | Rata-rata dan simpangan baku metrik pada fold validasi silang. Akurasi tidak sama dengan sensitivitas stunting; baca juga *balanced accuracy* dan recall. |
| `confusion_matrix_data600.png` | Jumlah prediksi benar dan salah untuk dua kelas. Baris adalah kelas aktual dan kolom adalah kelas prediksi. Kiri atas = *true negative* (Normal diprediksi Normal), kanan atas = *false positive*, kiri bawah = *false negative* (Stunting terlewat), kanan bawah = *true positive*. |
| `roc_curve_data600.png` | Sensitivitas versus *false positive rate* pada berbagai ambang. ROC-AUC merangkum kemampuan pemisahan kelas; garis diagonal menunjukkan prediksi acak. |
| `precision_recall_curve_data600.png` | Presisi versus recall kelas Stunting saat ambang berubah. Angka yang diberi label PR-AUC pada grafik dihitung dengan `average_precision_score`; ukuran ini berguna ketika distribusi kelas tidak seimbang. |
| `heatmap_correlation_data600_id.png` | Korelasi antar delapan fitur hasil transformasi dan target, dengan label Bahasa Indonesia. Merah menunjukkan korelasi positif, biru negatif, dan intensitas menunjukkan besar korelasi. |
| `heatmap_correlation_data600_en.png` | Heatmap yang sama dengan label Bahasa Inggris. Dua kategori dari satu variabel dapat tampil sebagai baris berbeda setelah *one-hot encoding*. |
| `shap_beeswarm_data600_id.png` | Sebaran kontribusi SHAP tiap fitur pada data uji, berlabel Bahasa Indonesia. Setiap titik mewakili satu observasi; posisi horizontal menunjukkan arah dan besar kontribusi, warna menunjukkan nilai fitur (merah tinggi, biru rendah). |
| `shap_beeswarm_data600_en.png` | Grafik SHAP yang sama dengan label Bahasa Inggris. Peringkat fitur menggambarkan pengaruh pada prediksi model, bukan bukti kausal. |
