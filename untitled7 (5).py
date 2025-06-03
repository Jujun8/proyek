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
def load_and_process_data(url):
    df = pd.read_csv(url)
    df['content'] = df['content'].astype(str) # Pastikan content adalah string

    # Tambahkan fitur numerik awal
    df['review_length'] = df['content'].str.len()
    df['word_count'] = df['content'].str.split().str.len()

    # Fitur Sentimen (Dummy berdasarkan 'score')
    # score 1,2 -> negatif (-0.5), 3 -> netral (0.0), 4,5 -> positif (0.5)
    sentiment_scores = []
    for score_val in df['score']:
        if score_val <= 2:
            sentiment_scores.append(-0.5)
        elif score_val == 3:
            sentiment_scores.append(0.0)
        else:
            sentiment_scores.append(0.5)
    df['sentiment_score'] = sentiment_scores

    # Fitur Penyebutan Kata Kunci (Sederhana)
    df['mentions_cod'] = df['content'].str.lower().str.contains('cod').astype(int)
    df['mentions_ongkir'] = df['content'].str.lower().str.contains('ongkir').astype(int)
    df['mentions_kecewa'] = df['content'].str.lower().str.contains('kecewa').astype(int)

    # Fitur dari Tanggal ('at')
    try:
        df['at'] = pd.to_datetime(df['at'])
        df['review_hour'] = df['at'].dt.hour
        df['review_day_of_week'] = df['at'].dt.dayofweek  # Senin=0, Minggu=6
    except Exception as e:
        st.warning(f"Error processing date column 'at': {e}. Date features will not be available.")
        df['review_hour'] = 0 # Default value jika error
        df['review_day_of_week'] = 0 # Default value jika error

    return df

# Load dan proses data
url = "https://github.com/Jujun8/proyek/blob/main/data%20proyek.csv"
df_original_raw = pd.read_csv(url) # Simpan dataframe mentah asli untuk tampilan awal
df = load_and_process_data(url) # DataFrame yang sudah diproses dengan fitur baru

# Pilih kolom numerik SETELAH semua fitur ditambahkan
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df_numerical = df[numerical_cols].copy()
df_numerical.dropna(inplace=True)

# Sidebar menu
menu = st.sidebar.selectbox("📁 Navigasi", ["Halaman Awal", "Model", "Prediksi"])

# ====================== HALAMAN AWAL ======================
if menu == "Halaman Awal":
    st.title("📊 Dashboard Data Proyek")

    st.subheader("Dataset Asli (Mentah)")
    st.dataframe(df_original_raw.head()) # Tampilkan data asli sebelum ada fitur tambahan

    st.subheader("Dataset Setelah Penambahan Fitur")
    st.dataframe(df.head())

    st.subheader("Karakteristik Data (Statistik Deskriptif untuk Kolom Numerik)")
    if not df[numerical_cols].empty: # Gunakan df karena numerical_cols berasal dari df
        st.dataframe(df[numerical_cols].describe())
    else:
        st.info("Tidak ada kolom numerik untuk ditampilkan statistiknya.")

    st.subheader("Visualisasi Data")
    valid_numerical_cols_for_viz = [col for col in numerical_cols if col in df.columns and df[col].nunique() > 1] # Hanya kolom dengan >1 nilai unik

    if len(valid_numerical_cols_for_viz) >= 1:
        viz_type = st.radio("Pilih Jenis Visualisasi", ["Scatter Plot", "Histogram", "Boxplot"])

        if viz_type == "Scatter Plot":
            if len(valid_numerical_cols_for_viz) >= 2:
                col1 = st.selectbox("Pilih fitur X", valid_numerical_cols_for_viz, index=0, key="scatter_x")
                col2_options = [col for col in valid_numerical_cols_for_viz if col != col1]
                if col2_options:
                    col2_default_index = 0
                    if 'score' in col2_options and col1 != 'score':
                         col2_default_index = col2_options.index('score')
                    elif len(col2_options) > 1 and 'sentiment_score' in col2_options and col1 != 'sentiment_score':
                         col2_default_index = col2_options.index('sentiment_score')


                    col2 = st.selectbox("Pilih fitur Y", col2_options, index=col2_default_index, key="scatter_y")
                    fig, ax = plt.subplots()
                    sns.scatterplot(data=df, x=col1, y=col2, hue='score' if 'score' in df.columns else None, alpha=0.7, ax=ax) # Tambahkan hue
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
            if 'sentiment_score' in valid_numerical_cols_for_viz:
                default_hist_col_index = valid_numerical_cols_for_viz.index('sentiment_score')
            elif 'score' in valid_numerical_cols_for_viz:
                default_hist_col_index = valid_numerical_cols_for_viz.index('score')

            selected_col_hist = st.selectbox("Pilih fitur", valid_numerical_cols_for_viz, index=default_hist_col_index, key="hist_select")
            fig, ax = plt.subplots()
            sns.histplot(df[selected_col_hist].dropna(), kde=True, bins=20, ax=ax)
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
            sns.boxplot(x=df[selected_col_box].dropna(), ax=ax)
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
            scaled_data = scaler.fit_transform(df_numerical)
        except ValueError as ve:
            st.error(f"Error saat scaling data: {ve}")
            st.error("Ini mungkin terjadi jika df_numerical hanya memiliki satu baris setelah dropna atau jika semua nilai dalam satu kolom adalah sama.")
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
        
        df_numerical_clustered = df_numerical.copy() # Bekerja dengan df_numerical yang sudah di-dropna
        df_numerical_clustered['Cluster'] = kmeans.fit_predict(scaled_data)

        st.session_state.kmeans_model = kmeans
        st.session_state.scaler_model = scaler
        # Simpan kolom yang BENAR-BENAR digunakan untuk training df_numerical (bukan numerical_cols dari df awal)
        st.session_state.numerical_cols_model = df_numerical.columns.tolist()


        # Inisialisasi kolom Cluster di df dengan nilai default
        df_with_clusters = df.copy() # Mulai dari df yang sudah diproses
        df_with_clusters['Cluster'] = -1 # Atau np.nan
        # Update kolom Cluster di df_with_clusters berdasarkan index dari df_numerical_clustered
        df_with_clusters.loc[df_numerical_clustered.index, 'Cluster'] = df_numerical_clustered['Cluster']

        st.subheader(f"🧾 Hasil Klastering dengan {n_clusters_selected} Klaster")
        st.dataframe(df_with_clusters[df_with_clusters['Cluster'] != -1].head())
        st.write(f"Menampilkan {len(df_with_clusters[df_with_clusters['Cluster'] != -1])} baris yang berhasil diklaster.")


        st.subheader("Statistik Tiap Klaster (Berdasarkan Fitur Numerik Asli yang Digunakan Model)")
        for cluster_id in range(n_clusters_selected):
            st.markdown(f"#### 📌 Statistik Cluster {cluster_id}")
            # Ambil data dari df_numerical_clustered untuk statistik
            # dan hanya kolom yang digunakan untuk model (df_numerical.columns)
            cluster_data_for_stats = df_numerical_clustered[df_numerical_clustered['Cluster'] == cluster_id][st.session_state.numerical_cols_model]
            if not cluster_data_for_stats.empty:
                st.dataframe(cluster_data_for_stats.describe())
            else:
                st.write("Tidak ada data untuk klaster ini.")
        
        # Visualisasi sederhana hasil cluster (jika memungkinkan)
        if len(st.session_state.numerical_cols_model) >= 2:
            st.subheader("Visualisasi Klaster (Contoh menggunakan 2 fitur pertama)")
            feat1_cluster_viz = st.session_state.numerical_cols_model[0]
            feat2_cluster_viz = st.session_state.numerical_cols_model[1]
            
            fig_cluster, ax_cluster = plt.subplots()
            sns.scatterplot(data=df_numerical_clustered, x=feat1_cluster_viz, y=feat2_cluster_viz, hue='Cluster', palette='viridis', ax=ax_cluster)
            ax_cluster.set_title(f'Klaster berdasarkan {feat1_cluster_viz} dan {feat2_cluster_viz}')
            st.pyplot(fig_cluster)


# ====================== HALAMAN PREDIKSI ======================
elif menu == "Prediksi":
    st.title("🔮 Prediksi Cluster untuk Data Baru")

    if 'kmeans_model' not in st.session_state or 'scaler_model' not in st.session_state or 'numerical_cols_model' not in st.session_state:
        st.warning("Model belum dilatih. Silakan ke halaman 'Model' terlebih dahulu untuk melatih model.")
    else:
        kmeans_trained = st.session_state.kmeans_model
        scaler_trained = st.session_state.scaler_model
        model_numerical_cols = st.session_state.numerical_cols_model # Ini adalah kolom dari df_numerical

        st.subheader("Masukkan Teks Ulasan Baru dan Skor:")
        input_content = st.text_area("Masukkan teks ulasan:", height=100, key="input_content_pred")
        input_score = st.number_input("Masukkan skor (1-5):", min_value=1, max_value=5, value=3, step=1, key="input_score_pred")

        # Tombol Prediksi
        if st.button("Prediksi Klaster", key="predict_button_new_data"):
            if not input_content:
                st.error("Teks ulasan tidak boleh kosong.")
            else:
                # Buat DataFrame sementara untuk data baru
                new_data = pd.DataFrame([{
                    'content': input_content,
                    'score': input_score,
                    'at': pd.Timestamp.now() # Tambahkan 'at' agar fitur tanggal bisa dihitung
                }])

                # Lakukan feature engineering pada data baru (mirip dengan load_and_process_data)
                new_data['content'] = new_data['content'].astype(str)
                new_data['review_length'] = new_data['content'].str.len()
                new_data['word_count'] = new_data['content'].str.split().str.len()

                if new_data['score'].iloc[0] <= 2:
                    new_data['sentiment_score'] = -0.5
                elif new_data['score'].iloc[0] == 3:
                    new_data['sentiment_score'] = 0.0
                else:
                    new_data['sentiment_score'] = 0.5

                new_data['mentions_cod'] = new_data['content'].str.lower().str.contains('cod').astype(int)
                new_data['mentions_ongkir'] = new_data['content'].str.lower().str.contains('ongkir').astype(int)
                new_data['mentions_kecewa'] = new_data['content'].str.lower().str.contains('kecewa').astype(int)

                new_data['at'] = pd.to_datetime(new_data['at']) # Sudah Timestamp, tapi pastikan
                new_data['review_hour'] = new_data['at'].dt.hour
                new_data['review_day_of_week'] = new_data['at'].dt.dayofweek

                # Pilih hanya kolom yang digunakan oleh model
                # Pastikan urutan kolom sama dengan saat training
                try:
                    input_df_features = new_data[model_numerical_cols]
                except KeyError as e:
                    st.error(f"Kolom yang hilang untuk prediksi: {e}. Pastikan semua fitur yang dibutuhkan model ada.")
                    st.error(f"Model membutuhkan kolom: {model_numerical_cols}")
                    st.error(f"Fitur yang dihasilkan dari input: {new_data.columns.tolist()}")
                    st.stop()


                # Scaling data input
                try:
                    input_scaled = scaler_trained.transform(input_df_features)
                    
                    # Prediksi cluster
                    cluster_pred = kmeans_trained.predict(input_scaled)[0]
                    st.success(f"✅ Data baru diprediksi termasuk ke dalam Cluster: {cluster_pred}")

                    st.subheader("Fitur yang Digunakan untuk Prediksi:")
                    st.dataframe(input_df_features)

                except ValueError as e:
                     st.error(f"Error saat scaling atau prediksi: {e}")
                     st.error("Pastikan input data baru memiliki format yang benar dan scaler telah dilatih dengan benar.")
                except Exception as e:
                    st.error(f"Terjadi kesalahan tak terduga saat prediksi: {e}")
