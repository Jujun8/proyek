import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Konfigurasi halaman
st.set_page_config(page_title="Clustering App", layout="wide")

# Load data
url = "https://raw.githubusercontent.com/Jujun8/sansan/main/data%20proyek.csv"
df = pd.read_csv(url)

# Sidebar menu
menu = st.sidebar.selectbox("Pilih Halaman", ["Dashboard", "Model", "Prediksi"])

# Halaman Dashboard
if menu == "Dashboard":
    st.title("📊 Dashboard Data Proyek")
    st.subheader("Tampilan Data")
    st.dataframe(df)

# Halaman Model
elif menu == "Model":
    st.title("🤖 Model Clustering (K-Means)")

    # Seleksi kolom numerik
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df_numerical = df[numerical_cols].dropna()

    if df_numerical.empty:
        st.warning("Tidak ada data numerik yang dapat digunakan untuk klastering.")
    else:
        # Standardisasi
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_numerical)

        # Tentukan jumlah cluster optimal dengan Elbow Method
        inertia = []
        k_range = range(1, 11)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(scaled_data)
            inertia.append(kmeans.inertia_)

        # Plot Elbow
        st.subheader("Elbow Method")
        fig, ax = plt.subplots()
        ax.plot(k_range, inertia, marker='o')
        ax.set_title('Elbow Method')
        ax.set_xlabel('Jumlah Klaster (k)')
        ax.set_ylabel('Inertia')
        st.pyplot(fig)

        # Pilih jumlah klaster
        n_clusters = st.slider("Pilih jumlah klaster", 2, 10, 3)

        # Jalankan klastering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df_numerical['Cluster'] = kmeans.fit_predict(scaled_data)

        # Gabungkan ke data asli
        df['Cluster'] = -1
        df.loc[df_numerical.index, 'Cluster'] = df_numerical['Cluster']

        st.subheader("Hasil Klastering")
        st.dataframe(df)

        # Statistik per klaster
        for cluster_id in range(n_clusters):
            st.subheader(f"📌 Statistik Cluster {cluster_id}")
            st.dataframe(df[df['Cluster'] == cluster_id].describe())

# Halaman Prediksi
elif menu == "Prediksi":
    st.title("🔮 Prediksi (Coming Soon)")
    st.info("Fitur prediksi akan ditambahkan di masa depan.")
