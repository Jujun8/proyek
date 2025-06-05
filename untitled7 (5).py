import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression # Contoh model klasifikasi
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
import csv
import re # Untuk pembersihan teks dasar

# Konfigurasi halaman
st.set_page_config(page_title="Sentiment Analysis App", layout="wide")

# Fungsi untuk membersihkan teks dasar
def simple_text_cleaner(text):
    if not isinstance(text, str):
        return ""
    text = text.lower() # Lowercase
    text = re.sub(r'[^\w\s]', '', text) # Hapus tanda baca
    text = re.sub(r'\d+', '', text) # Hapus angka
    return text.strip()

# Fungsi untuk memuat dan memproses data (agar bisa di-cache)
@st.cache_data
def load_data(url):
    try:
        df = pd.read_csv(url, engine='python', quoting=csv.QUOTE_MINIMAL, on_bad_lines='skip')
    except pd.errors.ParserError as pe:
        st.error(f"Pandas ParserError saat membaca CSV: {pe}")
        st.error("Mencoba membaca dengan parameter berbeda...")
        try:
            df = pd.read_csv(url, engine='python', quoting=csv.QUOTE_NONE, escapechar='\\', on_bad_lines='skip')
        except Exception as e:
            st.error(f"Gagal membaca CSV bahkan dengan parameter berbeda: {e}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Terjadi kesalahan umum saat memuat data: {e}")
        return pd.DataFrame()

    if 'content' in df.columns:
        df['content'] = df['content'].astype(str).fillna('') # Pastikan string dan isi NaN
        df['cleaned_content'] = df['content'].apply(simple_text_cleaner)
        df['review_length'] = df['cleaned_content'].str.len()
        df['word_count'] = df['cleaned_content'].str.split().str.len()
    else:
        st.warning("Kolom 'content' tidak ditemukan. Fitur teks tidak akan diproses.")
        df['cleaned_content'] = ""
        df['review_length'] = 0
        df['word_count'] = 0

    if 'score' not in df.columns:
        st.error("Kolom 'score' tidak ditemukan. Tidak dapat membuat target sentimen.")
        df['Sentiment'] = 'Unknown'
    else:
        df['score'] = pd.to_numeric(df['score'], errors='coerce')
        df.dropna(subset=['score'], inplace=True) # Hapus baris jika score NaN setelah konversi
        
        # Membuat target sentimen: 1,2 -> Negatif (0), 4,5 -> Positif (1)
        # Skor 3 bisa diabaikan atau dikategorikan (misal, netral)
        # Untuk binary classification, kita akan abaikan skor 3 atau assign
        conditions = [
            (df['score'] <= 2),
            (df['score'] >= 4)
        ]
        choices = ['Negatif', 'Positif']
        df['Sentiment'] = np.select(conditions, choices, default='Netral')
        # Filter hanya untuk Positif dan Negatif untuk model binary
        df = df[df['Sentiment'].isin(['Positif', 'Negatif'])]
        if df.empty:
            st.error("Tidak ada data valid untuk sentimen Positif/Negatif setelah pemrosesan skor.")
            return pd.DataFrame()
        df['Sentiment_Label'] = df['Sentiment'].map({'Positif': 1, 'Negatif': 0})


    # Hapus baris dengan 'cleaned_content' yang kosong setelah pembersihan, karena ini penting untuk TF-IDF
    df = df[df['cleaned_content'].str.strip() != '']
    if df.empty:
        st.error("Tidak ada data dengan konten teks yang valid setelah pembersihan.")
        return pd.DataFrame()

    return df

# Load data
url = "https://raw.githubusercontent.com/Jujun8/sansan/main/data%20proyek.csv"

# Menyimpan dataframe mentah asli untuk ditampilkan
try:
    df_original_raw = pd.read_csv(url, engine='python', quoting=csv.QUOTE_MINIMAL, on_bad_lines='skip')
except pd.errors.ParserError:
    try:
        df_original_raw = pd.read_csv(url, engine='python', quoting=csv.QUOTE_NONE, escapechar='\\', on_bad_lines='skip')
    except Exception:
        df_original_raw = pd.DataFrame()
except Exception:
    df_original_raw = pd.DataFrame()


df_processed = load_data(url)

if df_processed.empty:
    st.error("Gagal memuat atau memproses data. Aplikasi tidak dapat melanjutkan.")
    st.stop()

# Kolom numerik yang mungkin masih berguna (selain yang akan dibuat oleh TF-IDF)
numerical_cols_original = ['review_length', 'word_count']
# Pastikan kolom ini ada
numerical_cols_for_viz = [col for col in numerical_cols_original if col in df_processed.columns]


# Sidebar menu
menu_options = ["Halaman Awal", "Model Sentimen", "Prediksi Sentimen", "Informasi Aplikasi"]
menu = st.sidebar.selectbox("📁 Navigasi", menu_options)

# ====================== HALAMAN AWAL ======================
if menu == "Halaman Awal":
    st.title("📊 Dashboard Data Ulasan Pengguna")

    st.subheader("Dataset Asli (Mentah)")
    if not df_original_raw.empty:
        st.markdown(f"Menampilkan **{len(df_original_raw)}** baris data asli.")
        st.dataframe(df_original_raw.head())
    else:
        st.warning("Gagal menampilkan dataset asli mentah.")

    st.subheader("Dataset Setelah Pemrosesan dan Pembuatan Fitur")
    if not df_processed.empty:
        st.markdown(f"Menampilkan **{len(df_processed)}** baris data yang telah diproses (hanya ulasan Positif/Negatif).")
        st.dataframe(df_processed[['content', 'score', 'cleaned_content', 'review_length', 'word_count', 'Sentiment']].head())
    else:
        st.warning("Gagal menampilkan dataset yang diproses.")

    st.subheader("Distribusi Sentimen")
    if 'Sentiment' in df_processed.columns and not df_processed.empty:
        fig_sentiment, ax_sentiment = plt.subplots()
        sns.countplot(x='Sentiment', data=df_processed, ax=ax_sentiment, palette={'Positif': 'green', 'Negatif': 'red'})
        ax_sentiment.set_title("Distribusi Sentimen Ulasan")
        st.pyplot(fig_sentiment)
    else:
        st.info("Kolom 'Sentiment' tidak tersedia atau data kosong.")

    st.subheader("Karakteristik Data (Statistik Deskriptif untuk Fitur Tambahan)")
    if not df_processed.empty and numerical_cols_for_viz and not df_processed[numerical_cols_for_viz].empty:
        st.dataframe(df_processed[numerical_cols_for_viz + ['score']].describe()) # Tambahkan score jika ada
    else:
        st.info("Tidak ada kolom numerik ('review_length', 'word_count') untuk ditampilkan statistiknya atau data gagal dimuat.")

    st.subheader("Visualisasi Data Tambahan")
    if not df_processed.empty and numerical_cols_for_viz:
        viz_type = st.radio("Pilih Jenis Visualisasi", ["Histogram", "Boxplot"], key="viz_radio_home")

        if viz_type == "Histogram":
            selected_col_hist = st.selectbox("Pilih fitur", numerical_cols_for_viz, key="hist_select_home")
            fig, ax = plt.subplots()
            sns.histplot(data=df_processed, x=selected_col_hist, hue='Sentiment' if 'Sentiment' in df_processed.columns else None, kde=True, bins=30, ax=ax)
            ax.set_title(f'Distribusi {selected_col_hist}')
            st.pyplot(fig)

        elif viz_type == "Boxplot":
            selected_col_box = st.selectbox("Pilih fitur", numerical_cols_for_viz, key="box_select_home")
            fig, ax = plt.subplots()
            sns.boxplot(data=df_processed, x='Sentiment' if 'Sentiment' in df_processed.columns else None, y=selected_col_box, ax=ax)
            ax.set_title(f'Boxplot {selected_col_box} berdasarkan Sentimen')
            st.pyplot(fig)
    else:
        st.info("Tidak cukup fitur numerik untuk divisualisasikan.")


# ====================== HALAMAN MODEL SENTIMEN ======================
elif menu == "Model Sentimen":
    st.title("🤖 Model Klasifikasi Sentimen")

    if df_processed.empty or 'cleaned_content' not in df_processed.columns or 'Sentiment_Label' not in df_processed.columns:
        st.warning("Data yang diproses tidak lengkap atau tidak ada. Tidak dapat melatih model.")
        st.stop()
    
    if df_processed['cleaned_content'].isnull().any() or df_processed['cleaned_content'].eq('').any():
        st.warning("Ada nilai kosong atau string kosong di 'cleaned_content'. Model mungkin tidak optimal. Baris dengan konten kosong telah dihapus.")
        df_model_data = df_processed.dropna(subset=['cleaned_content'])
        df_model_data = df_model_data[df_model_data['cleaned_content'].str.strip() != '']
    else:
        df_model_data = df_processed.copy()

    if df_model_data.empty:
        st.error("Tidak ada data yang valid untuk melatih model setelah filter tambahan.")
        st.stop()

    X = df_model_data['cleaned_content']
    y = df_model_data['Sentiment_Label']

    if len(X) < 2 or len(y.unique()) < 2:
        st.error(f"Tidak cukup data atau variasi kelas untuk melatih model. Jumlah data: {len(X)}, Jumlah kelas unik: {len(y.unique())}")
        st.stop()
        
    # Split data
    # Stratify y untuk memastikan proporsi kelas sama di train dan test set, jika memungkinkan
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y if len(y.unique()) > 1 and y.value_counts().min() >=2 else None
        )
    except ValueError as e:
        st.warning(f"Tidak bisa melakukan stratify karena salah satu kelas mungkin terlalu sedikit: {e}. Melanjutkan tanpa stratify.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


    # Definisikan preprocessor untuk teks (TF-IDF)
    # Tidak ada fitur numerik lain yang akan digabung dengan TF-IDF di pipeline ini untuk kesederhanaan,
    # tapi bisa ditambahkan menggunakan ColumnTransformer jika diinginkan.
    text_processor = TfidfVectorizer(max_features=3000, ngram_range=(1,2)) # max_features dan ngram_range bisa di-tune

    # Buat pipeline: TF-IDF -> Classifier
    # Modelnya Logistic Regression untuk contoh
    model_pipeline = Pipeline([
        ('tfidf', text_processor),
        ('classifier', LogisticRegression(random_state=42, solver='liblinear', C=1.0)) # C bisa di-tune
    ])

    st.subheader("⚙️ Pelatihan Model")
    with st.spinner("Melatih model klasifikasi sentimen..."):
        try:
            model_pipeline.fit(X_train, y_train)
            st.session_state.sentiment_model_pipeline = model_pipeline
            st.session_state.X_test_sentiment = X_test # Simpan untuk evaluasi jika diperlukan
            st.session_state.y_test_sentiment = y_test
            st.success("Model berhasil dilatih!")
        except ValueError as e:
            st.error(f"Error saat melatih model: {e}")
            st.error("Ini bisa terjadi jika vocabulary kosong setelah preprocessing (misalnya, semua kata adalah stopwords atau teks sangat pendek).")
            st.stop()


    if 'sentiment_model_pipeline' in st.session_state:
        st.subheader("📊 Evaluasi Model pada Data Uji")
        model_trained = st.session_state.sentiment_model_pipeline
        X_test_eval = st.session_state.X_test_sentiment
        y_test_eval = st.session_state.y_test_sentiment

        try:
            y_pred = model_trained.predict(X_test_eval)
            accuracy = accuracy_score(y_test_eval, y_pred)
            
            st.write(f"Akurasi Model: **{accuracy:.2f}**")

            st.text("Laporan Klasifikasi:")
            # classification_report butuh target_names
            target_names = ['Negatif (0)', 'Positif (1)'] # Sesuai mapping Sentimen_Label
            report = classification_report(y_test_eval, y_pred, target_names=target_names, output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(report).transpose())

            st.text("Matriks Konfusi:")
            cm = confusion_matrix(y_test_eval, y_pred)
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names, ax=ax_cm)
            ax_cm.set_xlabel("Prediksi")
            ax_cm.set_ylabel("Aktual")
            st.pyplot(fig_cm)

        except Exception as e:
            st.error(f"Error saat evaluasi model: {e}")
            st.error("Ini bisa terjadi jika data uji tidak dapat diproses oleh model yang dilatih.")

# ====================== HALAMAN PREDIKSI SENTIMEN ======================
elif menu == "Prediksi Sentimen":
    st.title("🔮 Prediksi Sentimen untuk Komentar Baru")

    if 'sentiment_model_pipeline' not in st.session_state:
        st.warning("Model belum dilatih. Silakan ke halaman 'Model Sentimen' terlebih dahulu untuk melatih model.")
    else:
        model_to_predict = st.session_state.sentiment_model_pipeline
        
        st.subheader("Masukkan Komentar:")
        user_comment_input = st.text_area("Ketik atau tempel komentar di sini...", height=150, key="comment_input_pred")

        if st.button("Prediksi Sentimen", key="predict_button_sentiment"):
            if not user_comment_input.strip():
                st.error("Harap masukkan komentar untuk diprediksi.")
            else:
                # Preprocess input text sama seperti saat training
                cleaned_comment = simple_text_cleaner(user_comment_input)
                
                if not cleaned_comment.strip():
                    st.error("Komentar menjadi kosong setelah pembersihan dasar. Tidak dapat diprediksi.")
                else:
                    try:
                        # Prediksi menggunakan pipeline yang sudah dilatih
                        # Pipeline akan menangani TF-IDF transform
                        prediction_proba = model_to_predict.predict_proba([cleaned_comment])[0]
                        prediction_label = model_to_predict.predict([cleaned_comment])[0]

                        sentiment_map = {1: 'Positif', 0: 'Negatif'}
                        predicted_sentiment_text = sentiment_map.get(prediction_label, "Tidak diketahui")

                        st.subheader("Hasil Prediksi:")
                        if predicted_sentiment_text == 'Positif':
                            st.success(f"Sentimen: **{predicted_sentiment_text}**")
                        elif predicted_sentiment_text == 'Negatif':
                            st.error(f"Sentimen: **{predicted_sentiment_text}**")
                        else:
                            st.info(f"Sentimen: **{predicted_sentiment_text}**")
                        
                        st.write("Probabilitas:")
                        st.write(f"- Negatif: {prediction_proba[0]:.2%}")
                        st.write(f"- Positif: {prediction_proba[1]:.2%}")
                        
                        st.markdown("---")
                        st.write("Komentar yang Dianalisis (setelah pembersihan dasar):")
                        st.text(cleaned_comment)

                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat prediksi: {e}")
                        st.error("Ini bisa terjadi jika model tidak dapat memproses input. Pastikan model dilatih dengan benar.")

# ====================== HALAMAN INFORMASI APLIKASI ======================
elif menu == "Informasi Aplikasi":
    st.title("ℹ️ Informasi Aplikasi Analisis Sentimen")
    st.markdown("---")

    st.header("Tujuan Aplikasi")
    st.write("""
    Aplikasi ini dibangun untuk melakukan analisis sentimen pada dataset ulasan pengguna. 
    Tujuannya adalah untuk mengklasifikasikan ulasan sebagai 'Positif' atau 'Negatif' 
    berdasarkan konten teksnya. Ini dapat membantu memahami opini publik atau feedback pelanggan.
    """)

    st.header("Data yang Digunakan")
    st.write(f"""
    Dataset yang digunakan dalam aplikasi ini di ambil dari Kaggle (Anda bisa sebutkan nama datasetnya jika spesifik).
    Dataset ini berisi ulasan pengguna, termasuk skor yang diberikan dan konten ulasan.
    Sentimen ('Positif'/'Negatif') ditentukan berdasarkan kolom 'score':
    - Skor 1-2: Negatif
    - Skor 4-5: Positif
    - Skor 3: Dianggap Netral dan saat ini tidak dimasukkan dalam pelatihan model biner.
    """)
    st.write("""
    Fitur tambahan yang dibuat dari data asli adalah:
    - **`cleaned_content`**: Konten ulasan setelah pembersihan dasar (lowercase, hapus tanda baca & angka).
    - **`review_length`**: Panjang karakter dari `cleaned_content`.
    - **`word_count`**: Jumlah kata dalam `cleaned_content`.
    Fitur utama untuk model sentimen adalah representasi TF-IDF dari `cleaned_content`.
    """)

    st.header("Metode Analisis Sentimen")
    st.write("""
    Metode yang digunakan adalah **Klasifikasi Teks supervised learning**.
    1.  **Preprocessing Teks**: Teks ulasan dibersihkan.
    2.  **Vektorisasi Teks**: `cleaned_content` diubah menjadi fitur numerik menggunakan TF-IDF (Term Frequency-Inverse Document Frequency). Ini menangkap pentingnya kata dalam dokumen relatif terhadap seluruh korpus.
    3.  **Model Klasifikasi**: Model `Logistic Regression` dilatih pada fitur TF-IDF untuk membedakan antara ulasan positif dan negatif.
    4.  **Evaluasi**: Kinerja model diukur menggunakan metrik seperti akurasi, laporan klasifikasi (presisi, recall, F1-score), dan matriks konfusi pada data uji yang tidak terlihat saat pelatihan.
    """)

    st.header("Cara Menggunakan Aplikasi")
    st.write("""
    1.  **Navigasi**: Gunakan menu dropdown di sidebar kiri untuk berpindah antar halaman.
    2.  **Halaman Awal**: Menampilkan sampel dataset asli dan yang telah diproses, distribusi sentimen, statistik deskriptif, 
        serta visualisasi fitur tambahan.
    3.  **Model Sentimen**: 
        - Halaman ini melatih model klasifikasi sentimen secara otomatis ketika diakses.
        - Menampilkan metrik evaluasi kinerja model pada data uji, termasuk akurasi, laporan klasifikasi, dan matriks konfusi.
    4.  **Prediksi Sentimen**: 
        - Memungkinkan Anda memasukkan teks ulasan baru.
        - Aplikasi akan memprediksi apakah sentimen ulasan tersebut 'Positif' atau 'Negatif' menggunakan model yang telah dilatih.
    5.  **Informasi Aplikasi**: Halaman ini yang sedang Anda lihat, berisi penjelasan mengenai aplikasi.
    """)

    st.header("Tentang Proyek Ini")
    st.write("""
    Aplikasi ini adalah contoh bagaimana teknik pemrosesan bahasa alami (NLP) dan machine learning 
    dapat diterapkan untuk analisis sentimen.
    """)
    st.markdown("---")
    st.caption("Versi Aplikasi: 2.0.0 (Sentiment Analysis)")
