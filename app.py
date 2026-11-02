from pathlib import Path
import io
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ============================================================
# KONFIGURASI APLIKASI
# ============================================================
st.set_page_config(
    page_title="Global Earthquake Clustering",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌍 Global Earthquake Clustering Dashboard")
st.caption(
    "Analisis dan visualisasi pengelompokan data gempa bumi "
    "menggunakan algoritma K-Means."
)

FEATURES = ["latitude", "longitude", "depth", "mag"]
RANDOM_STATE = 42
DEFAULT_DATA_PATH = Path(__file__).parent / "global_earthquakes_seismic.csv"


# ============================================================
# FUNGSI PEMUATAN DAN PEMBERSIHAN DATA
# ============================================================
@st.cache_data(show_spinner="Membaca dataset...")
def load_data(file_bytes=None, file_name=None):
    if file_bytes is not None:
        data = pd.read_csv(io.BytesIO(file_bytes))
    else:
        if not DEFAULT_DATA_PATH.exists():
            return None
        data = pd.read_csv(DEFAULT_DATA_PATH)

    return data


def prepare_data(raw):
    data = raw.copy()

    if "type" in data.columns:
        data = data[
            data["type"].astype(str).str.lower().eq("earthquake")
        ].copy()

    missing_features = [col for col in FEATURES if col not in data.columns]
    if missing_features:
        raise ValueError(
            "Kolom fitur berikut tidak ditemukan: "
            + ", ".join(missing_features)
        )

    for col in FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    before = len(data)
    missing_by_feature = data[FEATURES].isna().sum()
    data = data.dropna(subset=FEATURES).drop_duplicates().copy()

    valid_geo = (
        data["latitude"].between(-90, 90)
        & data["longitude"].between(-180, 180)
    )
    invalid_geo = int((~valid_geo).sum())
    data = data.loc[valid_geo].copy()

    cleaning_info = {
        "rows_before_cleaning": before,
        "rows_after_cleaning": len(data),
        "rows_removed": before - len(data),
        "invalid_coordinates": invalid_geo,
        "missing_by_feature": missing_by_feature
    }

    return data, cleaning_info


# ============================================================
# FUNGSI ANALISIS
# ============================================================
@st.cache_data(show_spinner="Menghitung evaluasi jumlah cluster...")
def evaluate_k(X_scaled, k_min=2, k_max=8, sample_size=3000):
    results = []

    for k in range(k_min, k_max + 1):
        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=20
        )
        labels = model.fit_predict(X_scaled)

        score = silhouette_score(
            X_scaled,
            labels,
            sample_size=min(sample_size, len(X_scaled)),
            random_state=RANDOM_STATE
        )

        results.append({
            "K": k,
            "Inertia": model.inertia_,
            "Silhouette Score": score
        })

    return pd.DataFrame(results)


@st.cache_data(show_spinner="Menjalankan K-Means...")
def run_kmeans(X_scaled, k, sample_size=3000):
    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=20
    )
    labels = model.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        labels,
        sample_size=min(sample_size, len(X_scaled)),
        random_state=RANDOM_STATE
    )

    return labels, model.inertia_, score, model.cluster_centers_


def calculate_outlier_report(data):
    rows = []
    for feature in FEATURES:
        q1 = data[feature].quantile(0.25)
        q3 = data[feature].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        flagged = ~data[feature].between(lower, upper)

        rows.append({
            "Fitur": feature,
            "Batas bawah IQR": lower,
            "Batas atas IQR": upper,
            "Jumlah terindikasi": int(flagged.sum()),
            "Persentase (%)": round(flagged.mean() * 100, 2)
        })

    return pd.DataFrame(rows)


def eta_squared(data, label_col, feature):
    overall_mean = data[feature].mean()
    total_ss = ((data[feature] - overall_mean) ** 2).sum()

    if total_ss == 0:
        return 0.0

    between_ss = 0.0
    for _, group in data.groupby(label_col):
        between_ss += len(group) * (group[feature].mean() - overall_mean) ** 2

    return float(between_ss / total_ss)


# ============================================================
# SIDEBAR - SUMBER DATA DAN PARAMETER
# ============================================================
st.sidebar.header("Pengaturan Analisis")

uploaded_file = st.sidebar.file_uploader(
    "Unggah dataset CSV",
    type=["csv"],
    help="Unggah global_earthquakes_seismic.csv atau dataset dengan kolom fitur yang sesuai."
)

if uploaded_file is not None:
    raw_data = load_data(uploaded_file.getvalue(), uploaded_file.name)
    data_source = uploaded_file.name
else:
    raw_data = load_data()
    data_source = DEFAULT_DATA_PATH.name

if raw_data is None:
    st.warning(
        "Dataset belum ditemukan. Letakkan file "
        "`global_earthquakes_seismic.csv` satu folder dengan `app.py`, "
        "atau unggah CSV melalui sidebar."
    )
    st.stop()

try:
    df, cleaning_info = prepare_data(raw_data)
except Exception as exc:
    st.error(f"Dataset tidak dapat diproses: {exc}")
    st.stop()

if len(df) < 10:
    st.error("Jumlah data valid terlalu sedikit untuk analisis K-Means.")
    st.stop()

st.sidebar.caption(f"Sumber data: {data_source}")
st.sidebar.divider()

k_value = st.sidebar.slider(
    "Jumlah cluster (K)",
    min_value=2,
    max_value=8,
    value=4,
    step=1,
    help="Model K-Means akan dihitung ulang ketika nilai K diubah."
)

st.sidebar.subheader("Fitur Scatter Plot")
x_feature = st.sidebar.selectbox(
    "Fitur sumbu X",
    FEATURES,
    index=0
)
y_feature = st.sidebar.selectbox(
    "Fitur sumbu Y",
    FEATURES,
    index=1
)

if x_feature == y_feature:
    st.sidebar.warning("Pilih dua fitur yang berbeda untuk scatter plot.")

st.sidebar.subheader("Filter Data")
mag_min = float(df["mag"].min())
mag_max = float(df["mag"].max())
depth_min = float(df["depth"].min())
depth_max = float(df["depth"].max())

mag_range = st.sidebar.slider(
    "Rentang magnitudo",
    min_value=mag_min,
    max_value=mag_max,
    value=(mag_min, mag_max)
)
depth_range = st.sidebar.slider(
    "Rentang kedalaman (km)",
    min_value=depth_min,
    max_value=depth_max,
    value=(depth_min, depth_max)
)

filtered_df = df[
    df["mag"].between(mag_range[0], mag_range[1])
    & df["depth"].between(depth_range[0], depth_range[1])
].copy()

if len(filtered_df) < k_value + 1:
    st.error("Data setelah filter terlalu sedikit untuk jumlah cluster yang dipilih.")
    st.stop()

# Scaling mengikuti data yang sedang dianalisis/terfilter.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(filtered_df[FEATURES])

evaluation_df = evaluate_k(X_scaled, 2, 8, 3000)
labels, inertia, silhouette, centers_scaled = run_kmeans(
    X_scaled, k_value, 3000
)

result_df = filtered_df.copy()
result_df["cluster"] = labels.astype(int)

pca = PCA(n_components=2)
pca_values = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(
    pca_values,
    columns=["PC1", "PC2"],
    index=result_df.index
)
pca_df["cluster"] = labels.astype(str)

centers_original = pd.DataFrame(
    scaler.inverse_transform(centers_scaled),
    columns=FEATURES
)
centers_original.index.name = "cluster"


# ============================================================
# NAVIGASI TAB
# ============================================================
tabs = st.tabs([
    "Overview",
    "EDA",
    "Preprocessing",
    "K-Means & Evaluasi",
    "Visualisasi Cluster",
    "Interpretasi",
    "Unduh Hasil"
])


# ============================================================
# TAB 1 - OVERVIEW
# ============================================================
with tabs[0]:
    st.header("Ringkasan Dataset")
    st.write(
        "Dashboard ini mengelompokkan kejadian gempa bumi berdasarkan "
        "latitude, longitude, depth, dan magnitude menggunakan K-Means."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Data dianalisis", f"{len(result_df):,}")
    c2.metric("Fitur clustering", len(FEATURES))
    c3.metric("Jumlah cluster", k_value)
    c4.metric("Silhouette Score", f"{silhouette:.4f}")

    st.subheader("Informasi Dataset")
    info1, info2 = st.columns(2)
    with info1:
        st.write(f"**Sumber file:** {data_source}")
        st.write(f"**Jumlah baris dataset asli:** {len(raw_data):,}")
        st.write(f"**Jumlah baris setelah pembersihan:** {len(df):,}")
        st.write(f"**Jumlah kolom:** {raw_data.shape[1]}")
    with info2:
        st.write("**Fitur yang digunakan:**")
        st.write(", ".join(FEATURES))
        st.write(f"**Baris yang dikeluarkan saat cleaning:** {cleaning_info['rows_removed']:,}")
        st.write(f"**Koordinat tidak valid:** {cleaning_info['invalid_coordinates']:,}")

    st.subheader("Cuplikan Dataset")
    st.dataframe(result_df.head(20), use_container_width=True)

    st.subheader("Statistik Deskriptif")
    st.dataframe(result_df[FEATURES].describe().T, use_container_width=True)

    st.info(
        "Interpretasi: ringkasan di atas menunjukkan jumlah data yang benar-benar "
        "digunakan setelah filter dan pembersihan. Ketika filter magnitudo atau "
        "kedalaman diubah, jumlah data dan model juga berubah."
    )


# ============================================================
# TAB 2 - EDA
# ============================================================
with tabs[1]:
    st.header("Exploratory Data Analysis (EDA)")

    st.subheader("Distribusi Fitur")
    selected_distribution = st.selectbox(
        "Pilih fitur untuk melihat distribusi",
        FEATURES,
        key="distribution_feature"
    )

    fig_hist = px.histogram(
        filtered_df,
        x=selected_distribution,
        marginal="box",
        nbins=40,
        title=f"Distribusi {selected_distribution}"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    q1 = filtered_df[selected_distribution].quantile(0.25)
    median = filtered_df[selected_distribution].median()
    q3 = filtered_df[selected_distribution].quantile(0.75)
    st.markdown(
        f"**Analisis grafik:** median {selected_distribution} adalah "
        f"**{median:.3f}**, kuartil pertama **{q1:.3f}**, dan kuartil ketiga "
        f"**{q3:.3f}**. Perhatikan bentuk histogram untuk menilai kemencengan "
        "dan konsentrasi data."
    )

    st.subheader("Boxplot Seluruh Fitur")
    box_df = filtered_df[FEATURES].melt(
        var_name="Fitur",
        value_name="Nilai"
    )
    fig_box = px.box(
        box_df,
        x="Fitur",
        y="Nilai",
        color="Fitur",
        title="Boxplot Fitur Gempa"
    )
    st.plotly_chart(fig_box, use_container_width=True)
    st.markdown(
        "**Analisis grafik:** titik di luar whisker merupakan kandidat nilai ekstrem "
        "menurut aturan boxplot. Nilai tersebut tidak otomatis salah dan perlu "
        "ditinjau dalam konteks karakteristik gempa."
    )

    st.subheader("Correlation Heatmap")
    corr = filtered_df[FEATURES].corr()
    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Korelasi Pearson Antarfitur"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    max_corr = corr.where(~np.eye(len(corr), dtype=bool)).abs().stack()
    if not max_corr.empty:
        pair = max_corr.idxmax()
        st.markdown(
            f"**Analisis grafik:** hubungan linear absolut paling besar pada "
            f"pasangan **{pair[0]} dan {pair[1]}**, dengan korelasi "
            f"**{corr.loc[pair[0], pair[1]]:.3f}**. Korelasi tidak menunjukkan "
            "hubungan sebab-akibat."
        )

    st.subheader("Pairplot / Hubungan Antarfitur")
    st.caption(
        "Untuk menjaga performa dashboard, scatter matrix dibuat dari maksimal "
        "1.000 sampel acak."
    )
    pair_sample = filtered_df[FEATURES].sample(
        n=min(1000, len(filtered_df)),
        random_state=RANDOM_STATE
    )
    fig_pair = px.scatter_matrix(
        pair_sample,
        dimensions=FEATURES,
        title="Scatter Matrix Antarfitur"
    )
    fig_pair.update_traces(diagonal_visible=False, showupperhalf=False)
    st.plotly_chart(fig_pair, use_container_width=True)
    st.markdown(
        "**Analisis grafik:** scatter matrix memperlihatkan pola hubungan "
        "berpasangan dan kemungkinan kelompok data. Pola visual ini bersifat "
        "eksploratif dan perlu dikonfirmasi melalui hasil evaluasi clustering."
    )


# ============================================================
# TAB 3 - PREPROCESSING
# ============================================================
with tabs[2]:
    st.header("Data Preprocessing")

    st.subheader("Pemeriksaan Missing Values")
    missing_table = raw_data.isna().sum().to_frame("Jumlah Missing Values")
    missing_table["Persentase (%)"] = (
        raw_data.isna().mean() * 100
    ).round(2)
    st.dataframe(missing_table, use_container_width=True)

    st.subheader("Ringkasan Pembersihan")
    p1, p2, p3 = st.columns(3)
    p1.metric("Baris sebelum cleaning", f"{cleaning_info['rows_before_cleaning']:,}")
    p2.metric("Baris setelah cleaning", f"{cleaning_info['rows_after_cleaning']:,}")
    p3.metric("Baris dikeluarkan", f"{cleaning_info['rows_removed']:,}")

    st.write("Missing values pada fitur terpilih sebelum pembersihan:")
    st.dataframe(
        cleaning_info["missing_by_feature"].to_frame("Jumlah"),
        use_container_width=True
    )

    st.subheader("Identifikasi Outlier dengan IQR")
    outlier_report = calculate_outlier_report(filtered_df)
    st.dataframe(outlier_report, use_container_width=True)
    st.warning(
        "Outlier dipertahankan pada model utama. IQR digunakan untuk identifikasi, "
        "bukan untuk menyatakan bahwa observasi ekstrem merupakan kesalahan."
    )

    st.subheader("Hasil StandardScaler")
    scaled_df = pd.DataFrame(
        X_scaled,
        columns=FEATURES,
        index=filtered_df.index
    )
    st.dataframe(scaled_df.describe().T.round(4), use_container_width=True)
    st.markdown(
        "**Analisis preprocessing:** StandardScaler membuat fitur berada pada "
        "skala yang sebanding. Ini penting karena K-Means menghitung jarak "
        "Euclidean dan sensitif terhadap perbedaan skala."
    )


# ============================================================
# TAB 4 - MODEL DAN EVALUASI
# ============================================================
with tabs[3]:
    st.header("K-Means dan Evaluasi Model")
    st.write(
        "Pilih K melalui slider di sidebar. Model, label cluster, inertia, "
        "dan Silhouette Score akan dihitung ulang mengikuti nilai K."
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("K terpilih", k_value)
    m2.metric("Inertia", f"{inertia:,.2f}")
    m3.metric("Silhouette Score", f"{silhouette:.4f}")

    st.subheader("Elbow Method")
    fig_elbow = px.line(
        evaluation_df,
        x="K",
        y="Inertia",
        markers=True,
        title="Elbow Method: Inertia terhadap Jumlah Cluster"
    )
    st.plotly_chart(fig_elbow, use_container_width=True)
    st.markdown(
        "**Analisis grafik:** inertia menunjukkan jumlah kuadrat jarak data "
        "terhadap centroid. Nilai cenderung turun ketika K bertambah. Titik "
        "siku dicari pada bagian ketika penurunan mulai melandai; inertia "
        "terkecil saja bukan alasan memilih K."
    )

    st.subheader("Silhouette Score")
    fig_sil = px.line(
        evaluation_df,
        x="K",
        y="Silhouette Score",
        markers=True,
        title="Silhouette Score untuk K = 2 sampai 8"
    )
    st.plotly_chart(fig_sil, use_container_width=True)

    best_row = evaluation_df.loc[evaluation_df["Silhouette Score"].idxmax()]
    st.markdown(
        f"**Analisis grafik:** pada rentang K=2–8, skor tertinggi diperoleh "
        f"pada K={int(best_row['K'])} dengan skor {best_row['Silhouette Score']:.4f}. "
        f"Model saat ini menggunakan K={k_value} dengan skor {silhouette:.4f}. "
        "Pemilihan akhir juga perlu mempertimbangkan Elbow dan interpretasi cluster."
    )

    st.subheader("Distribusi Anggota Cluster")
    counts = result_df["cluster"].value_counts().sort_index().reset_index()
    counts.columns = ["Cluster", "Jumlah Data"]
    fig_counts = px.bar(
        counts,
        x="Cluster",
        y="Jumlah Data",
        color="Cluster",
        title="Jumlah Anggota Setiap Cluster"
    )
    st.plotly_chart(fig_counts, use_container_width=True)
    st.dataframe(counts, use_container_width=True)


# ============================================================
# TAB 5 - VISUALISASI CLUSTER
# ============================================================
with tabs[4]:
    st.header("Visualisasi Hasil Clustering")

    if x_feature == y_feature:
        st.warning("Pilih dua fitur berbeda pada sidebar untuk menampilkan scatter plot.")
    else:
        fig_scatter = px.scatter(
            result_df,
            x=x_feature,
            y=y_feature,
            color=result_df["cluster"].astype(str),
            hover_data=FEATURES,
            opacity=0.7,
            title=f"Scatter Plot Cluster: {x_feature} vs {y_feature}",
            labels={"color": "Cluster"}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.markdown(
            f"**Analisis grafik:** grafik menampilkan distribusi data berdasarkan "
            f"{x_feature} dan {y_feature}. Warna menunjukkan label cluster hasil "
            "K-Means. Tumpang tindih pada dua fitur tidak selalu berarti cluster "
            "tidak terpisah pada keseluruhan empat dimensi."
        )

    st.subheader("Visualisasi PCA 2D")
    pca_plot = pca_df.copy()
    pca_plot["cluster"] = pca_plot["cluster"].astype(str)
    fig_pca = px.scatter(
        pca_plot,
        x="PC1",
        y="PC2",
        color="cluster",
        opacity=0.7,
        title="Cluster pada Ruang PCA"
    )
    st.plotly_chart(fig_pca, use_container_width=True)
    pca_variance = pca.explained_variance_ratio_.sum() * 100
    st.markdown(
        f"**Analisis grafik:** dua komponen PCA menjelaskan {pca_variance:.2f}% "
        "variasi data. PCA membantu melihat pola dalam dua dimensi, tetapi "
        "tidak menampilkan seluruh informasi dari empat fitur."
    )

    st.subheader("Sebaran Geografis")
    fig_geo = px.scatter_geo(
        result_df,
        lat="latitude",
        lon="longitude",
        color=result_df["cluster"].astype(str),
        hover_data=["depth", "mag"],
        projection="natural earth",
        opacity=0.65,
        title="Persebaran Gempa Berdasarkan Cluster"
    )
    st.plotly_chart(fig_geo, use_container_width=True)
    st.markdown(
        "**Analisis grafik:** visualisasi memperlihatkan lokasi koordinat kejadian "
        "gempa menurut cluster. Peta ini merupakan plot sebaran lokasi, bukan "
        "peta risiko atau batas zona tektonik."
    )


# ============================================================
# TAB 6 - INTERPRETASI CLUSTER
# ============================================================
with tabs[5]:
    st.header("Karakteristik dan Interpretasi Cluster")

    profile = result_df.groupby("cluster")[FEATURES].agg(
        ["mean", "median", "min", "max"]
    )
    profile.columns = [
        f"{feature}_{stat}" for feature, stat in profile.columns
    ]
    profile.insert(0, "jumlah_data", result_df.groupby("cluster").size())
    profile["persentase"] = (
        profile["jumlah_data"] / len(result_df) * 100
    ).round(2)

    st.subheader("Tabel Profil Cluster")
    st.dataframe(profile.round(3), use_container_width=True)

    st.subheader("Rata-rata Fitur per Cluster")
    means = result_df.groupby("cluster")[FEATURES].mean().reset_index()
    means_long = means.melt(
        id_vars="cluster",
        var_name="Fitur",
        value_name="Rata-rata"
    )
    fig_means = px.bar(
        means_long,
        x="Fitur",
        y="Rata-rata",
        color="cluster",
        barmode="group",
        title="Perbandingan Rata-rata Fitur Asli"
    )
    st.plotly_chart(fig_means, use_container_width=True)
    st.markdown(
        "**Analisis grafik:** bandingkan rata-rata fitur di dalam fitur yang sama. "
        "Jangan membandingkan tinggi nilai latitude dengan depth atau magnitude "
        "secara langsung karena satuannya berbeda."
    )

    st.subheader("Fitur yang Paling Membedakan Cluster")
    eta_table = pd.DataFrame({
        "Fitur": FEATURES,
        "Eta-squared": [
            eta_squared(result_df, "cluster", feature)
            for feature in FEATURES
        ]
    }).sort_values("Eta-squared", ascending=False)

    fig_eta = px.bar(
        eta_table,
        x="Fitur",
        y="Eta-squared",
        title="Ukuran Pemisahan Antarcluster per Fitur (Eta-squared)"
    )
    st.plotly_chart(fig_eta, use_container_width=True)
    st.dataframe(eta_table.round(4), use_container_width=True)

    leading_feature = eta_table.iloc[0]["Fitur"]
    st.markdown(
        f"**Analisis:** fitur dengan eta-squared terbesar pada hasil ini adalah "
        f"**{leading_feature}**. Ukuran ini menunjukkan proporsi variasi fitur "
        "yang terkait dengan perbedaan rata-rata antarcluster. Ini merupakan "
        "ukuran deskriptif, bukan feature importance bawaan K-Means atau bukti sebab-akibat."
    )

    st.subheader("Ringkasan Otomatis Setiap Cluster")
    for cluster_id, row in profile.iterrows():
        with st.expander(f"Cluster {cluster_id}"):
            st.write(
                f"Jumlah anggota: **{int(row['jumlah_data']):,}** "
                f"({row['persentase']:.2f}% dari data yang dianalisis)."
            )
            st.write(
                f"Rata-rata latitude: {row['latitude_mean']:.3f}; "
                f"longitude: {row['longitude_mean']:.3f}; "
                f"depth: {row['depth_mean']:.3f} km; "
                f"magnitude: {row['mag_mean']:.3f}."
            )
            st.write(
                "Gunakan median dan rentang pada tabel untuk melengkapi interpretasi, "
                "terutama jika distribusi fitur menceng atau memiliki nilai ekstrem."
            )


# ============================================================
# TAB 7 - UNDUH HASIL
# ============================================================
with tabs[6]:
    st.header("Unduh Hasil Analisis")

    scaled_export = pd.DataFrame(
        X_scaled,
        columns=FEATURES,
        index=result_df.index
    )
    scaled_export["cluster"] = labels

    summary_export = profile.copy()

    def csv_bytes(dataframe):
        return dataframe.to_csv(index=True).encode("utf-8-sig")

    st.download_button(
        "⬇️ Unduh data hasil clustering",
        data=result_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="data_hasil_clustering.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.download_button(
        "⬇️ Unduh data normalisasi",
        data=scaled_export.to_csv(index=False).encode("utf-8-sig"),
        file_name="data_normalisasi.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.download_button(
        "⬇️ Unduh ringkasan cluster",
        data=csv_bytes(summary_export),
        file_name="ringkasan_cluster.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.download_button(
        "⬇️ Unduh evaluasi jumlah K",
        data=evaluation_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="evaluasi_jumlah_cluster.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.caption(
        "File hasil mengikuti filter magnitudo, kedalaman, fitur, dan nilai K "
        "yang sedang aktif pada dashboard."
    )


# ============================================================
# CATATAN METODOLOGIS
# ============================================================
st.divider()
st.caption(
    "Catatan metodologis: K-Means menggunakan jarak Euclidean pada fitur "
    "yang distandarisasi. Koordinat latitude dan longitude diperlakukan sebagai "
    "variabel numerik biasa, bukan jarak geodesik. Hasil cluster bersifat "
    "eksploratif dan tidak boleh ditafsirkan sebagai klasifikasi resmi risiko gempa."
)
