import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
import io

# CSS Styling
st.markdown("""
<style>
            /* Ganti warna background seluruh halaman */
[data-testid="stAppViewContainer"] {
    background-color: #F0F8FF;  /* Contoh: warna biru muda */
}
/* Ubah warna font default di seluruh halaman */
[data-testid="stAppViewContainer"] {
    color: #1B2631 !important;  /* warna font gelap */
}
            /* Ubah warna font sidebar */
[data-testid="stSidebar"] {
    color: #154360 !important;
}
            /* Ubah warna font label dan teks di uploader file */
section[data-testid="stFileUploader"] label {
    color: #AED6F1 !important;
}
/* Opsional: warna background sidebar */
[data-testid="stSidebar"] {
    background-color: #D6EAF8;
}
            /* Supaya teks di header dan paragraf tidak putih */
h1, h2, h3, p, label, div, span {
    color: #154360 !important;
h1, h3 {
    color: #2E86C1;
    text-align: center;
}
footer {
    text-align: center;
    color: #999;
    margin-top: 30px;
}
.cluster-info {
    background-color: #D6EAF8;
    border-left: 5px solid #2980B9;
    padding: 15px;
    margin-top: 20px;
    font-family: sans-serif;
}

/* Gaya upload file custom */
section[data-testid="stFileUploader"] > div {
    background-color: #1C2833;
    border: 2px dashed #3498DB;
    border-radius: 10px;
    padding: 20px;
    transition: 0.3s ease;
}

section[data-testid="stFileUploader"]:hover > div {
    border-color: #5DADE2;
    background-color: #212F3D;
}

section[data-testid="stFileUploader"] label {
    font-size: 16px;
    font-weight: bold;
    color: #AED6F1;
}

section[data-testid="stFileUploader"] svg {
    stroke: #5DADE2;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.app-title {
    font-size: 36px;
    font-weight: bold;
    color: #154360;
    text-align: center;
    background: linear-gradient(to right, #D6EAF8, #AED6F1);
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 30px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    font-family: 'Segoe UI', sans-serif;
}
</style>

<div class='app-title'>
    SmartSeg: Aplikasi Segmentasi Pelanggan Cerdas Berbasis Machine Learning
</div>
""", unsafe_allow_html=True)

# Tab navigasi
tab1, tab2, tab3 = st.tabs(["📊 Segmentasi", "📂 Dataset", "ℹ️ Tentang"])

with tab1:
    st.header("Segmentasi Pelanggan")

    uploaded_file = st.file_uploader("Upload dataset CSV pelanggan:", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Preview Dataset")
        st.dataframe(df.head())

        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) < 2:
            st.error("Dataset harus punya minimal 2 kolom numerik.")
        else:
            # Atur index default kolom supaya tidak error jika hanya 2 kolom
            default_index_col1 = 0
            default_index_col2 = 1 if len(numeric_cols) > 1 else 0
            
            col1 = st.selectbox("Pilih fitur X", options=numeric_cols, index=default_index_col1)
            # Pilih kolom lain selain col1
            available_cols_for_col2 = [c for c in numeric_cols if c != col1]
            if not available_cols_for_col2:
                st.error("Tidak ada kolom lain selain kolom X yang bisa dipilih untuk fitur Y.")
            else:
                col2 = st.selectbox("Pilih fitur Y", options=available_cols_for_col2, index=default_index_col2-1 if default_index_col2>0 else 0)

                X = df[[col1, col2]].values
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                algo = st.selectbox("Pilih algoritma segmentasi", ["K-Means", "DBSCAN", "Hierarchical Clustering"])

                if algo == "K-Means":
                    n_clusters = st.slider("Jumlah cluster", 2, 10, 3)
                    model = KMeans(n_clusters=n_clusters, random_state=42)
                    labels = model.fit_predict(X_scaled)
                elif algo == "DBSCAN":
                    eps = st.slider("Nilai eps (radius)", 0.1, 5.0, 0.5)
                    min_samples = st.slider("Minimal samples", 3, 20, 5)
                    model = DBSCAN(eps=eps, min_samples=min_samples)
                    labels = model.fit_predict(X_scaled)
                else:  # Hierarchical
                    n_clusters = st.slider("Jumlah cluster", 2, 10, 3)
                    model = AgglomerativeClustering(n_clusters=n_clusters)
                    labels = model.fit_predict(X_scaled)

                # Visualisasi
                fig, ax = plt.subplots(figsize=(8, 5))

                # Untuk DBSCAN, label -1 artinya noise, kita buat warna khusus
                if algo == "DBSCAN":
                    palette = sns.color_palette("Set2", np.unique(labels).max() + 1)
                    colors = [palette[x] if x >= 0 else (0.5,0.5,0.5) for x in labels]
                    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=colors)
                else:
                    sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=labels, palette="Set2", ax=ax, legend='full')

                ax.set_title("Hasil Segmentasi")
                ax.set_xlabel(col1)
                ax.set_ylabel(col2)
                st.pyplot(fig)

                # Hitung jumlah cluster (DBSCAN ada noise -1 yang tidak dihitung cluster)
                unique_labels = set(labels)
                n_clusters_ = len(unique_labels) - (1 if -1 in unique_labels else 0)

                # Tambahkan label cluster ke DataFrame
                df["Cluster"] = labels
                st.markdown("<h3>Penjelasan Segmentasi</h3>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class='cluster-info'>
                Algoritma <b>{algo}</b> berhasil membagi pelanggan ke dalam <b>{n_clusters_}</b> cluster berdasarkan fitur <b>{col1}</b> dan <b>{col2}</b>.<br>
                Setiap cluster menandakan kelompok pelanggan dengan kemiripan karakteristik.
                </div>
                """, unsafe_allow_html=True)

                # Tampilkan hasil cluster
                st.subheader("Hasil Data + Cluster")
                st.dataframe(df)

                # Download hasil ke Excel
                buffer = io.BytesIO()
                df.to_excel(buffer, index=False, engine='openpyxl')
                st.download_button(
                    "📥 Download hasil ke Excel",
                    data=buffer.getvalue(),
                    file_name="hasil_segmentasi.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

with tab2:
    st.header("📂 Penjelasan Dataset")
    st.markdown("""
    Pastikan dataset memiliki:
    - Minimal 2 kolom numerik (misalnya: Umur, Total Belanja, Frekuensi Kunjungan)
    - Format file: .csv
    
    Contoh isi dataset:
    | Umur | Belanja | Kunjungan |
    |------|---------|-----------|
    | 25   | 500000  | 5         |
    | 32   | 1200000 | 10        |
    """)

with tab3:
    st.header("ℹ️ Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini dibuat untuk membantu bisnis dalam melakukan *segmentasi pelanggan* dengan berbagai algoritma machine learning:

    | Algoritma               | Tujuan |
    |-------------------------|--------|
    | *K-Means*             | Mengelompokkan pelanggan berdasarkan kemiripan data numerik |
    | *DBSCAN*              | Mengelompokkan berdasarkan kepadatan, cocok untuk data tidak beraturan |
    | *Hierarchical Clustering* | Mengelompokkan secara bertingkat, cocok untuk visualisasi dendrogram |

    Dibuat oleh: *Kelompok 1*  
    Tahun: *2025*  
    """)

st.markdown("<footer>© 2025 - Segmentasi Pelanggan App by Kelompok 1</footer>", unsafe_allow_html=True)