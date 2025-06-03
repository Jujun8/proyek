import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

# Konfigurasi halaman
st.set_page_config(page_title="Clustering App", layout="wide")

# Fungsi untuk memuat dan memproses data (agar bisa di-cache)
@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    # Tambahkan fitur numerik baru (HANYA YANG AWAL)
    df['content'] = df['content'].astype(str) # Pastikan content adalah string
    df['review_length'] = df['content'].str.len()
    df['word_count'] = df['content'].str.split().str.len()
    return df

# Load data
url = "https://raw.githubusercontent.com/Jujun8/sansan/main/data%20proyek.csv"
df_original_raw = pd.read_csv(url) # DataFrame asli mentah dari CSV
df_processed = load_data(url) # DataFrame dengan fitur 'review_length' dan 'word_count'

# Pilih kolom numerik dari DataFrame yang sudah diproses
numerical_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
df_numerical = df_processed[numerical_cols].copy()
df_numerical.dropna(inplace=True) # Hapus NaN dari data numerik yang akan digunakan untuk clustering

# Sidebar menu
menu = st.sidebar.selectbox("📁 Navigasi", ["Halaman Awal", "Model", "Prediksi"])

# ====================== HALAMAN AWAL ======================
if menu == "Halaman Awal":
    st.title("📊 Dashboard Data Proyek")

    st.subheader("Dataset Asli (Mentah)")
    st.dataframe(df_original_raw.head()) # Tampilkan head data asli

    st.subheader("Dataset Setelah Penambahan Fitur Dasar ('review_length', 'word_count')")
    st.dataframe(df_processed.head()) # Tampilkan head data setelah penambahan fitur dasar

    st.subheader("Karakteristik Data (Statistik Deskriptif untuk Kolom Numerik)")
    # Gunakan numerical_cols yang didefinisikan dari df_processed
    if not df_processed[numerical_cols].empty:
        st.dataframe(df_processed[numerical_cols].describe())
    else:
        st.info("Tidak ada kolom numerik untuk ditampilkan statistiknya.")

    st.subheader("Visualisasi Data")
    # Filter numerical_cols yang ada di df_processed dan punya variasi
    valid_numerical_cols_for_viz = [col for col in numerical_cols if col in df_processed.columns and df_processed[col].nunique() > 1]

    if len(valid_numerical_cols_for_viz) >= 1:
        viz_type = st.radio("Pilih Jenis Visualisasi", ["Scatter Plot", "Histogram", "Boxplot"])

        if viz_type == "Scatter Plot":
            if len(valid_numerical_cols_for_viz) >= 2:
                col1_default_index = 0
                if 'score' in valid_numerical_cols_for_viz:
                    col1_default_index = valid_numerical_cols_for_viz.index('score')
                col1 = st.selectbox("Pilih fitur X", valid_numerical_cols_for_viz, index=col1_default_index, key="scatter_x")

                col2_options = [col for col in valid_numerical_cols_for_viz if col != col1]
                if col2_options:
                    col2_default_index = 0
                    if 'review_length' in col2_options:
                         col2_default_index = col2_options.index('review_length')
                    elif len(col2_options) > 0:
                        col2_default_index = 0 # fallback

                    col2 = st.selectbox("Pilih fitur Y", col2_options, index=col2_default_index, key="scatter_y")
                    fig, ax = plt.subplots()
                    # Gunakan df_processed untuk plotting
                    sns.scatterplot(data=df_processed, x=col1, y=col2, hue='score' if 'score' in df_processed.columns else None, alpha=0.7, ax=ax)
                    ax.set_xlabel(col1)
                    ax.set_ylabel(col2)
                    ax.set_title(f'Scatter Plot: {col1} vs {col2}')
                    st.pyplot(fig)
                else:
                    st.warning("Tidak cukup fitur berbeda untuk scatter plot Y.")
            else:
                st.warning("Scatter plot membutuhkan minimal 2 fitur numerik.")

        elif viz_type == "Histogram":
            default_hist_col_index = 0
            if 'score' in valid_numerical_cols_for_viz:
                default_hist_col_index = valid_numerical_cols_for_viz.index('score')
            selected_col_hist = st.selectbox("Pilih fitur", valid_numerical_cols_for_viz, index=default_hist_col_index, key="hist_select")
            fig, ax = plt.subplots()
            sns.histplot(df_processed[selected_col_hist].dropna(), kde=True, bins=20, ax=ax)
            ax.set_title(f'Distribusi Nilai {selected_col_hist}')
            st.pyplot(fig)

        elif viz_type == "Boxplot":
            default_box_col_index = 0
            if 'review_length' in valid_numerical_cols_for_viz:
                default_box_col_index = valid_numerical_cols_for_viz.index('review_length')
            elif 'score' in valid_numerical_cols_for_viz:
                 default_box_col_index = valid_numerical_cols_for_viz.index('score')
            selected_col_box = st.selectbox("Pilih fitur", valid_numerical_cols_for_viz, index=default_box_col_index, key="box_select")
            fig, ax = plt.subplots()
            sns.boxplot(x=df_processed[selected_col_box].dropna(), ax=ax)
            ax.set_title(f'Boxplot {selected_col_box}')
            st.pyplot(fig)
    else:
        st.info("Tidak cukup fitur numerik yang bervariasi untuk divisualisasikan.")

# ====================== HALAMAN MODEL ======================
elif menu == "Model":
    st.title("🤖 Model Clustering (K-Means)")

    if df_numerical.empty or len(df_numerical.columns) == 0:
        st.warning("Tidak ada data numerik yang dapat digunakan untuk klastering setelah menghapus NaN atau tidak ada kolom numerik yang tersisa.")
    else:
        scaler = StandardScaler()
        try:
            scaled_data = scaler.fit_transform(df_numerical) # df_numerical adalah hasil dari df_processed[numerical_cols].dropna()
        except ValueError as ve:
            st.error(f"Error saat scaling data: {ve}")
            st.stop()

        st.subheader("📐 Elbow Method untuk Menentukan Jumlah Klaster Optimal")
        inertia = []
        k_range = range(1, 11)
        with st.spinner("Menghitung Elbow Method..."):
            for k_val in k_range:
                kmeans_elbow = KMeans(n_clusters=k_val, random_state=42, n_init='auto')
                kmeans_elbow.fit(scaled_data)
                inertia.append(kmeans_elbow.inertia_)

        fig_elbow, ax_elbow = plt.subplots()
        ax_elbow.plot(k_range, inertia, marker='o')
        ax_elbow.set_xlabel("Jumlah Klaster (k)")
        ax_elbow.set_ylabel("Inertia")
        ax_elbow.set_title("Elbow Method")
        st.pyplot(fig_elbow)

        if 'n_clusters_model' not in st.session_state:
            st.session_state.n_clusters_model = 3

        st.session_state.n_clusters_model = st.slider(
            "Pilih jumlah klaster (k)", 2, 10, st.session_state.n_clusters_model
        )
        n_clusters_selected = st.session_state.n_clusters_model

        kmeans = KMeans(n_clusters=n_clusters_selected, random_state=42, n_init='auto')
        
        df_numerical_clustered = df_numerical.copy() # df_numerical sudah di-dropna dan hanya berisi kolom numerik
        df_numerical_clustered['Cluster'] = kmeans.fit_predict(scaled_data)

        st.session_state.kmeans_model = kmeans
        st.session_state.scaler_model = scaler
        st.session_state.numerical_cols_model = df_numerical.columns.tolist() # Kolom yang digunakan untuk model

        # Tambah hasil cluster ke dataframe df_processed untuk ditampilkan
        df_processed_with_clusters = df_processed.copy()
        df_processed_with_clusters['Cluster'] = -1 # Default
        df_processed_with_clusters.loc[df_numerical_clustered.index, 'Cluster'] = df_numerical_clustered['Cluster']

        st.subheader(f"🧾 Hasil Klastering dengan {n_clusters_selected} Klaster")
        # Tampilkan dari df_processed_with_clusters yang barisnya berhasil diklaster
        st.dataframe(df_processed_with_clusters[df_processed_with_clusters['Cluster'] != -1].head())
        st.write(f"Menampilkan {len(df_processed_with_clusters[df_processed_with_clusters['Cluster'] != -1])} baris yang berhasil diklaster.")

        st.subheader("Statistik Tiap Klaster (Berdasarkan Fitur Numerik yang Digunakan Model)")
        for cluster_id in range(n_clusters_selected):
            st.markdown(f"#### 📌 Statistik Cluster {cluster_id}")
            cluster_data_for_stats = df_numerical_clustered[df_numerical_clustered['Cluster'] == cluster_id][st.session_state.numerical_cols_model]
            if not cluster_data_for_stats.empty:
                st.dataframe(cluster_data_for_stats.describe())
            else:
                st.write("Tidak ada data untuk klaster ini.")
        
        if len(st.session_state.numerical_cols_model) >= 2:
            st.subheader("Visualisasi Klaster (Contoh menggunakan 2 fitur pertama dari model)")
            feat1_cluster_viz = st.session_state.numerical_cols_model[0]
            feat2_cluster_viz = st.session_state.numerical_cols_model[1]
            
            fig_cluster_viz, ax_cluster_viz = plt.subplots()
            sns.scatterplot(data=df_numerical_clustered, x=feat1_cluster_viz, y=feat2_cluster_viz, hue='Cluster', palette='viridis', ax=ax_cluster_viz)
            ax_cluster_viz.set_title(f'Klaster berdasarkan {feat1_cluster_viz} dan {feat2_cluster_viz}')
            st.pyplot(fig_cluster_viz)

# ====================== HALAMAN PREDIKSI ======================
elif menu == "Prediksi":
    st.title("🔮 Prediksi Cluster untuk Data Baru")

    if 'kmeans_model' not in st.session_state or 'scaler_model' not in st.session_state or 'numerical_cols_model' not in st.session_state:
        st.warning("Model belum dilatih. Silakan ke halaman 'Model' terlebih dahulu untuk melatih model.")
    else:
        kmeans_trained = st.session_state.kmeans_model
        scaler_trained = st.session_state.scaler_model
        model_numerical_cols = st.session_state.numerical_cols_model # Ini adalah kolom dari df_numerical

        st.subheader("Masukkan Nilai Fitur untuk Data Baru:")
        st.markdown(f"Model dilatih menggunakan fitur berikut: `{'`, `'.join(model_numerical_cols)}`")

        input_data_pred = {} # Gunakan nama variabel berbeda untuk menghindari konflik
        # Ambil df_numerical yang terakhir digunakan untuk melatih model untuk nilai default
        # Ini diasumsikan df_numerical adalah state data yang benar untuk default
        df_for_defaults = df_numerical 

        for col_pred in model_numerical_cols:
            default_val_pred = 0.0
            if col_pred in df_for_defaults.columns:
                 default_val_pred = float(df_for_defaults[col_pred].mean())
            
            input_data_pred[col_pred] = st.number_input(
                f"Nilai untuk '{col_pred}'",
                value=default_val_pred,
                format="%.2f",
                key=f"input_pred_{col_pred}" # Key unik untuk input prediksi
            )

        if st.button("Prediksi Klaster", key="predict_button_manual_input"):
            try:
                # Buat DataFrame dari input_data dengan urutan kolom yang sama seperti saat training
                input_df_pred = pd.DataFrame([input_data_pred])[model_numerical_cols]
                
                input_scaled_pred = scaler_trained.transform(input_df_pred)
                
                cluster_pred_val = kmeans_trained.predict(input_scaled_pred)[0]
                st.success(f"✅ Data baru diprediksi termasuk ke dalam Cluster: {cluster_pred_val}")

                st.subheader("Fitur yang Dimasukkan untuk Prediksi:")
                st.dataframe(input_df_pred)

            except ValueError as e:
                st.error(f"Error saat scaling atau prediksi: {e}. Periksa apakah semua input numerik dan sesuai.")
            except Exception as e:
                st.error(f"Terjadi kesalahan tak terduga saat prediksi: {e}")
