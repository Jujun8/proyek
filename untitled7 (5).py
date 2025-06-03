import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Konfigurasi halaman
st.set_page_config(page_title="Clustering App", layout="wide")

# Load data
url = "https://raw.githubusercontent.com/Jujun8/sansan/main/data%20proyek.csv"
df = pd.read_csv(url)

# Tambahkan fitur numerik baru
df['review_length'] = df['content'].astype(str).str.len()
df['word_count'] = df['content'].astype(str).str.split().str.len()

# Pilih kolom numerik
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
df_numerical = df[numerical_cols].dropna()

# Sidebar menu
menu = st.sidebar.selectbox("📁 Navigasi", ["Halaman Awal", "Model", "Prediksi"])

# ====================== HALAMAN AWAL ======================
if menu == "Halaman Awal":
    st.title("📊 Dashboard Data Proyek")

    st.subheader("Dataset")
    st.dataframe(df)

    st.subheader("Karakteristik Data (Statistik Deskriptif)")
    st.dataframe(df[numerical_cols].describe())

    st.subheader("Visualisasi Data")

    if len(numerical_cols) >= 1:
        viz_type = st.radio("Pilih Jenis Visualisasi", ["Scatter Plot", "Histogram", "Boxplot"])

        if viz_type == "Scatter Plot" and len(numerical_cols) >= 2:
            col1 = st.selectbox("Pilih fitur X", numerical_cols, index=0)
            col2 = st.selectbox("Pilih fitur Y", numerical_cols, index=1)
            fig, ax = plt.subplots()
            ax.scatter(df[col1], df[col2], alpha=0.7)
            ax.set_xlabel(col1)
            ax.set_ylabel(col2)
            ax.set_title(f'Scatter Plot: {col1} vs {col2}')
            st.pyplot(fig)

        elif viz_type == "Histogram":
            selected_col = st.selectbox("Pilih fitur", numerical_cols)
            fig, ax = plt.subplots()
            sns.histplot(df[selected_col], kde=True, bins=10, ax=ax)
            ax.set_title(f'Distribusi Nilai {selected_col}')
            st.pyplot(fig)

        elif viz_type == "Boxplot":
            selected_col = st.selectbox("Pilih fitur", numerical_cols)
            fig, ax = plt.subplots()
            sns.boxplot(x=df[selected_col], ax=ax)
            ax.set_title(f'Boxplot {selected_col}')
            st.pyplot(fig)
    else:
        st.info("Tidak cukup fitur numerik untuk divisualisasikan.")

# ====================== HALAMAN MODEL ======================
elif menu == "Model":
    st.title("🤖 Model Clustering (K-Means)")

    if df_numerical.empty:
        st.warning("Tidak ada data numerik yang dapat digunakan untuk klastering.")
    else:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_numerical)

        # Elbow Method
        inertia = []
        k_range = range(1, 11)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(scaled_data)
            inertia.append(kmeans.inertia_)

        st.subheader("📐 Elbow Method")
        fig, ax = plt.subplots()
        ax.plot(k_range, inertia, marker='o')
        ax.set_xlabel("Jumlah Klaster (k)")
        ax.set_ylabel("Inertia")
        ax.set_title("Menentukan Jumlah Klaster Optimal")
        st.pyplot(fig)

        # Slider klaster
        n_clusters = st.slider("Pilih jumlah klaster", 2, 10, 3)

        # KMeans Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df_numerical['Cluster'] = kmeans.fit_predict(scaled_data)

        # Tambah hasil ke df
        df['Cluster'] = -1
        df.loc[df_numerical.index, 'Cluster'] = df_numerical['Cluster']

        st.subheader("🧾 Hasil Klastering")
        st.dataframe(df)

        # Statistik tiap cluster
        for cluster_id in range(n_clusters):
            st.subheader(f"📌 Statistik Cluster {cluster_id}")
            st.dataframe(df[df['Cluster'] == cluster_id][numerical_cols].describe())

# ====================== HALAMAN PREDIKSI ======================
elif menu == "Prediksi":
    st.title("🔮 Prediksi Cluster untuk Data Baru")

    if df_numerical.empty:
        st.warning("Tidak ada data numerik yang tersedia untuk pelatihan model.")
    else:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_numerical)

        # Default KMeans
        n_clusters = 3
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(scaled_data)

        st.subheader("Masukkan Nilai Fitur:")
        input_data = []
        for col in numerical_cols:
            value = st.number_input(f"{col}", value=float(df[col].mean()))
            input_data.append(value)

        if st.button("Prediksi Klaster"):
            input_scaled = scaler.transform([input_data])
            cluster = kmeans.predict(input_scaled)[0]
            st.success(f"✅ Data termasuk ke dalam Cluster: {cluster}")
