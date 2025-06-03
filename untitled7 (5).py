import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from datetime import datetime

# Konfigurasi halaman
st.set_page_config(page_title="Analisis Ulasan Pengguna", layout="wide")

# Load data
url = "https://raw.githubusercontent.com/Jujun8/sansan/main/data%20proyek.csv"
df = pd.read_csv(url)

# Parsing kolom tanggal
df['at'] = pd.to_datetime(df['at'], errors='coerce')

# Sidebar menu
menu = st.sidebar.selectbox("📁 Navigasi", ["Halaman Awal", "Statistik & Visualisasi", "Filter Data", "Tabel Interaktif"])

# Halaman Awal
if menu == "Halaman Awal":
    st.title("📊 Dashboard Ulasan Pengguna")
    st.subheader("Dataset")
    st.dataframe(df)

# Statistik & Visualisasi
elif menu == "Statistik & Visualisasi":
    st.title("📈 Statistik dan Visualisasi")

    st.subheader("1. Statistik Ringkas")
    st.metric("Jumlah Ulasan", len(df))
    st.metric("Skor Rata-rata", round(df['score'].mean(), 2))

    st.subheader("2. Distribusi Skor")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='score', palette='viridis', ax=ax)
    ax.set_title("Distribusi Skor Ulasan")
    st.pyplot(fig)

    st.subheader("3. Ulasan per Hari")
    reviews_per_day = df['at'].dt.date.value_counts().sort_index()
    fig, ax = plt.subplots()
    reviews_per_day.plot(kind='line', ax=ax)
    ax.set_title("Jumlah Ulasan per Hari")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Jumlah")
    st.pyplot(fig)

    st.subheader("4. Word Cloud Ulasan")
    text = " ".join(df['content'].dropna().tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

# Filter Data
elif menu == "Filter Data":
    st.title("🔍 Filter Ulasan")

    skor = st.slider("Pilih rentang skor", 1, 5, (1, 5))
    tanggal_mulai = st.date_input("Tanggal mulai", df['at'].min().date())
    tanggal_akhir = st.date_input("Tanggal akhir", df['at'].max().date())
    kata_kunci = st.text_input("Cari kata dalam ulasan")

    mask = (df['score'].between(skor[0], skor[1])) & \
           (df['at'].dt.date >= tanggal_mulai) & \
           (df['at'].dt.date <= tanggal_akhir)

    if kata_kunci:
        mask &= df['content'].str.contains(kata_kunci, case=False, na=False)

    st.subheader("Hasil Filter")
    st.dataframe(df[mask])

# Tabel Interaktif
elif menu == "Tabel Interaktif":
    st.title("📋 Tabel Interaktif")
    st.dataframe(df.sort_values(by="at", ascending=False))
