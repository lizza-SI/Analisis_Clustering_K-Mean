# Global Earthquake Clustering Dashboard

Dashboard interaktif untuk praktikum Analisis dan Visualisasi Clustering menggunakan K-Means.

## Fitur
- Ringkasan dataset dan statistik deskriptif
- EDA: histogram, boxplot, correlation heatmap, scatter matrix
- Pemeriksaan missing values, duplikasi, dan outlier IQR
- StandardScaler
- Elbow Method dan Silhouette Score
- K-Means dengan pilihan jumlah cluster (K)
- Scatter plot dengan pemilihan fitur
- PCA 2D dan peta sebaran geografis
- Profil cluster, eta-squared deskriptif, dan unduh CSV

## Struktur folder

```text
earthquake_clustering_streamlit/
├── app.py
├── requirements.txt
├── README.md
└── global_earthquakes_seismic.csv
```

## Instalasi dan menjalankan

1. Pastikan Python 3.10 atau lebih baru tersedia.
2. Buka terminal pada folder proyek.
3. Buat virtual environment (opsional tetapi disarankan):

   Windows:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

   macOS/Linux:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. Instal library:

   ```bash
   pip install -r requirements.txt
   ```

5. Letakkan `global_earthquakes_seismic.csv` di folder yang sama dengan `app.py`.
   Alternatifnya, unggah CSV melalui sidebar dashboard.

6. Jalankan:

   ```bash
   streamlit run app.py
   ```

7. Streamlit akan membuka dashboard di browser. Jika tidak terbuka otomatis, buka alamat lokal yang ditampilkan terminal (umumnya `http://localhost:8501`).

## Catatan metodologis

Fitur clustering utama adalah `latitude`, `longitude`, `depth`, dan `mag`. Model melakukan StandardScaler dan K-Means pada data yang sedang dipilih melalui filter. Karena K-Means menggunakan jarak Euclidean, latitude dan longitude diperlakukan sebagai variabel numerik biasa, bukan jarak geodesik. Hasil bersifat eksploratif, bukan klasifikasi resmi risiko gempa.

Silhouette Score dihitung menggunakan sampel maksimal 3.000 baris untuk menjaga performa. Angka tersebut merupakan estimasi berbasis sampel.

## GitHub dan Streamlit Community Cloud

- Buat repository GitHub baru.
- Unggah `app.py`, `requirements.txt`, `README.md`, dan dataset CSV (jika ukuran file sesuai kebijakan GitHub).
- Untuk dataset besar, jangan commit file data; gunakan fitur upload CSV pada dashboard atau penyimpanan data eksternal.
- Di Streamlit Community Cloud, pilih repository, branch, dan file utama `app.py`.
