import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np # Ditambahkan untuk menangani potensi NaN di input numerik

# Konfigurasi halaman
st.set_page_config(page_title="Clustering App", layout="wide")

# Fungsi untuk memuat dan memproses data (agar bisa di-cache)
@st.cache_data # Menggunakan cache agar data tidak dimuat ulang setiap interaksi
def load_data(url):
    df = pd.read_csv(url)
    # Tambahkan fitur numerik baru
    df['review_length'] = df['content'].astype(str).str.len()
    df['word_count'] = df['content'].astype(str).str.split().str.len()
    return df

# Load data
url = "https://raw.githubusercontent.com/Jujun8/sansan/main/data%20proyek.csv"
df_original = load_data(url) # Simpan dataframe asli
df = df_original.copy() # Bekerja dengan salinan agar df_original tetap utuh

# Pilih kolom numerik
# Pastikan ini dijalankan setelah df dimuat dan fitur baru ditambahkan
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist() # Lebih aman menggunakan np.number
df_numerical = df[numerical_cols].copy() # Buat salinan eksplisit
df_numerical.dropna(inplace=True) # Hapus NaN dari data numerik yang akan digunakan untuk clustering

# Sidebar menu
menu = st.sidebar.selectbox("📁 Navigasi", ["Halaman Awal", "Model", "Prediksi"])

# ====================== HALAMAN AWAL ======================
if menu == "Halaman Awal":
    st.title("📊 Dashboard Data Proyek")

    st.subheader("Dataset Asli")
    st.dataframe(df_original.head()) # Tampilkan head untuk performa jika data besar

    st.subheader("Karakteristik Data (Statistik Deskriptif untuk Kolom Numerik)")
    if not df[numerical_cols].empty:
        st.dataframe(df[numerical_cols].describe())
    else:
        st.info("Tidak ada kolom numerik untuk ditampilkan statistiknya.")


    st.subheader("Visualisasi Data")
    # Filter numerical_cols yang ada di df (jika ada perubahan dinamis)
    valid_numerical_cols_for_viz = [col for col in numerical_cols if col in df.columns]

    if len(valid_numerical_cols_for_viz) >= 1:
        viz_type = st.radio("Pilih Jenis Visualisasi", ["Scatter Plot", "Histogram", "Boxplot"])

        if viz_type == "Scatter Plot":
            if len(valid_numerical_cols_for_viz) >= 2:
                col1 = st.selectbox("Pilih fitur X", valid_numerical_cols_for_viz, index=0, key="scatter_x")
                col2_options = [col for col in valid_numerical_cols_for_viz if col != col1]
                if col2_options: # Pastikan ada pilihan untuk col2
                    col2 = st.selectbox("Pilih fitur Y", col2_options, index=0 if len(col2_options) > 0 else 0, key="scatter_y")
                    fig, ax = plt.subplots()
                    ax.scatter(df[col1], df[col2], alpha=0.7)
                    ax.set_xlabel(col1)
                    ax.set_ylabel(col2)
                    ax.set_title(f'Scatter Plot: {col1} vs {col2}')
                    st.pyplot(fig)
                else:
                    st.warning("Tidak cukup fitur berbeda untuk scatter plot Y.")
            else:
                st.warning("Scatter plot membutuhkan minimal 2 fitur numerik.")


        elif viz_type == "Histogram":
            selected_col_hist = st.selectbox("Pilih fitur", valid_numerical_cols_for_viz, key="hist_select")
            fig, ax = plt.subplots()
            sns.histplot(df[selected_col_hist].dropna(), kde=True, bins=20, ax=ax) # Tambah dropna() untuk plot
            ax.set_title(f'Distribusi Nilai {selected_col_hist}')
            st.pyplot(fig)

        elif viz_type == "Boxplot":
            selected_col_box = st.selectbox("Pilih fitur", valid_numerical_cols_for_viz, key="box_select")
            fig, ax = plt.subplots()
            sns.boxplot(x=df[selected_col_box].dropna(), ax=ax) # Tambah dropna() untuk plot
            ax.set_title(f'Boxplot {selected_col_box}')
            st.pyplot(fig)
    else:
        st.info("Tidak cukup fitur numerik untuk divisualisasikan.")

# ====================== HALAMAN MODEL ======================
elif menu == "Model":
    st.title("🤖 Model Clustering (K-Means)")

    if df_numerical.empty:
        st.warning("Tidak ada data numerik yang dapat digunakan untuk klastering setelah menghapus NaN.")
    else:
        # Inisialisasi scaler dan scaled_data di sini
        scaler = StandardScaler()
        # Pastikan df_numerical hanya berisi kolom yang benar-benar numerik dan tidak ada NaN
        # df_numerical sudah di .dropna() sebelumnya
        scaled_data = scaler.fit_transform(df_numerical)

        st.subheader("📐 Elbow Method untuk Menentukan Jumlah Klaster Optimal")
        inertia = []
        k_range = range(1, 11)
        with st.spinner("Menghitung Elbow Method..."):
            for k_val in k_range:
                kmeans_elbow = KMeans(n_clusters=k_val, random_state=42, n_init='auto') # n_init='auto' lebih modern
                kmeans_elbow.fit(scaled_data)
                inertia.append(kmeans_elbow.inertia_)

        fig_elbow, ax_elbow = plt.subplots()
        ax_elbow.plot(k_range, inertia, marker='o')
        ax_elbow.set_xlabel("Jumlah Klaster (k)")
        ax_elbow.set_ylabel("Inertia")
        ax_elbow.set_title("Elbow Method")
        st.pyplot(fig_elbow)

        # Slider klaster
        # Idealnya, simpan n_clusters di st.session_state jika ingin digunakan di halaman lain
        if 'n_clusters_model' not in st.session_state:
            st.session_state.n_clusters_model = 3 # Default value

        st.session_state.n_clusters_model = st.slider(
            "Pilih jumlah klaster (k)", 2, 10, st.session_state.n_clusters_model
        )
        n_clusters_selected = st.session_state.n_clusters_model

        # KMeans Clustering
        kmeans = KMeans(n_clusters=n_clusters_selected, random_state=42, n_init='auto')
        
        # Buat salinan dari df_numerical untuk menyimpan cluster agar tidak mempengaruhi df_numerical asli
        df_numerical_clustered = df_numerical.copy()
        df_numerical_clustered['Cluster'] = kmeans.fit_predict(scaled_data)

        # Simpan model dan scaler ke session_state untuk digunakan di halaman prediksi
        st.session_state.kmeans_model = kmeans
        st.session_state.scaler_model = scaler
        st.session_state.numerical_cols_model = df_numerical.columns.tolist() # Simpan kolom yg dipakai model


        # Tambah hasil cluster ke dataframe original (df)
        # Inisialisasi kolom cluster di df dengan nilai default (misal -1 atau NaN)
        df['Cluster'] = -1 # Atau np.nan jika lebih disukai
        # Update kolom Cluster di df berdasarkan index dari df_numerical_clustered
        df.loc[df_numerical_clustered.index, 'Cluster'] = df_numerical_clustered['Cluster']

        st.subheader(f"🧾 Hasil Klastering dengan {n_clusters_selected} Klaster")
        st.dataframe(df.head()) # Tampilkan head untuk performa
        st.write(f"Menampilkan {len(df[df['Cluster'] != -1])} baris yang berhasil diklaster (data non-NaN numerik).")


        st.subheader("Statistik Tiap Klaster (Berdasarkan Fitur Numerik Asli)")
        for cluster_id in range(n_clusters_selected):
            st.markdown(f"#### 📌 Statistik Cluster {cluster_id}")
            # Ambil data dari df_numerical_clustered untuk statistik, bukan df_numerical asli
            cluster_data = df_numerical_clustered[df_numerical_clustered['Cluster'] == cluster_id][df_numerical.columns] # exclude 'Cluster' column
            if not cluster_data.empty:
                st.dataframe(cluster_data.describe())
            else:
                st.write("Tidak ada data untuk klaster ini.")

# ====================== HALAMAN PREDIKSI ======================
elif menu == "Prediksi":
    st.title("🔮 Prediksi Cluster untuk Data Baru")

    # Cek apakah model sudah dilatih dari halaman "Model"
    if 'kmeans_model' not in st.session_state or 'scaler_model' not in st.session_state or 'numerical_cols_model' not in st.session_state:
        st.warning("Model belum dilatih. Silakan ke halaman 'Model' terlebih dahulu untuk melatih model.")
        st.info("Jika Anda sudah melatih model, pastikan slider jumlah klaster di halaman 'Model' sudah diatur dan proses klastering selesai.")
    else:
        # Ambil model, scaler, dan kolom numerik dari session_state
        kmeans_trained = st.session_state.kmeans_model
        scaler_trained = st.session_state.scaler_model
        model_numerical_cols = st.session_state.numerical_cols_model

        st.subheader("Masukkan Nilai Fitur untuk Data Baru:")
        st.markdown(f"Model dilatih menggunakan fitur berikut: `{'`, `'.join(model_numerical_cols)}`")

        input_data = {}
        for col in model_numerical_cols:
            # Ambil nilai rata-rata dari df_numerical asli (sebelum scaling) untuk default
            # Ini mengasumsikan df_numerical (yang digunakan untuk melatih scaler) masih tersedia
            # Jika df_numerical tidak selalu ada, mungkin lebih baik menyimpan means saat training
            default_val = 0.0
            if col in df_numerical.columns: # df_numerical dari scope global, yg sudah di dropna
                 default_val = float(df_numerical[col].mean())

            input_data[col] = st.number_input(
                f"Nilai untuk '{col}'",
                value=default_val,
                format="%.2f", # Format agar lebih rapi
                key=f"input_{col}"
            )

        if st.button("Prediksi Klaster", key="predict_button"):
            # Buat DataFrame dari input_data dengan urutan kolom yang sama seperti saat training
            try:
                input_df = pd.DataFrame([input_data])[model_numerical_cols]
                
                # Scaling data input menggunakan scaler yang sudah di-fit
                input_scaled = scaler_trained.transform(input_df)
                
                # Prediksi cluster
                cluster_pred = kmeans_trained.predict(input_scaled)[0]
                st.success(f"✅ Data baru diprediksi termasuk ke dalam Cluster: {cluster_pred}")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat prediksi: {e}")
                st.error("Pastikan semua nilai input valid dan model telah dilatih dengan benar.")
