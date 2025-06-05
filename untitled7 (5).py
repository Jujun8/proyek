import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.preprocessing import StandardScaler # Tidak digunakan lagi di versi sentimen ini
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression # Contoh model klasifikasi
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer # Tidak digunakan lagi
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
        
        conditions = [
            (df['score'] <= 2),
            (df['score'] >= 4)
        ]
        choices = ['Negatif', 'Positif']
        df['Sentiment'] = np.select(conditions, choices, default='Netral')
        df = df[df['Sentiment'].isin(['Positif', 'Negatif'])]
        if df.empty:
            st.error("Tidak ada data valid untuk sentimen Positif/Negatif setelah pemrosesan skor.")
            return pd.DataFrame()
        df['Sentiment_Label'] = df['Sentiment'].map({'Positif': 1, 'Negatif': 0})

    df = df[df['cleaned_content'].str.strip() != '']
    if df.empty:
        st.error("Tidak ada data dengan konten teks yang valid setelah pembersihan.")
        return pd.DataFrame()

    return df

# Load data
url = "https://raw.githubusercontent.com/Jujun8/sansan/main/data%20proyek.csv"

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

numerical_cols_original = ['review_length', 'word_count']
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
        st.dataframe(df_processed[numerical_cols_for_viz + (['score'] if 'score' in df_processed.columns else [])].describe())
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
    st.title("🤖 Model Performance: Klasifikasi Sentimen")

    if df_processed.empty or 'cleaned_content' not in df_processed.columns or 'Sentiment_Label' not in df_processed.columns:
        st.warning("Data yang diproses tidak lengkap atau tidak ada. Tidak dapat melatih model.")
        st.stop()
    
    df_model_data = df_processed.dropna(subset=['cleaned_content'])
    df_model_data = df_model_data[df_model_data['cleaned_content'].str.strip() != '']

    if df_model_data.empty:
        st.error("Tidak ada data yang valid untuk melatih model setelah filter tambahan.")
        st.stop()

    X = df_model_data['cleaned_content']
    y = df_model_data['Sentiment_Label']

    if len(X) < 2 or len(y.unique()) < 2:
        st.error(f"Tidak cukup data atau variasi kelas untuk melatih model. Jumlah data: {len(X)}, Jumlah kelas unik: {len(y.unique())}")
        st.stop()
        
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y if y.value_counts().min() >=2 else None
        )
    except ValueError as e:
        st.warning(f"Tidak bisa melakukan stratify karena salah satu kelas mungkin terlalu sedikit: {e}. Melanjutkan tanpa stratify.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # --- Model Selection (Placeholder for future model choice) ---
    # Untuk sekarang, kita hardcode Logistic Regression, tapi Anda bisa menambahkan pilihan model di sini
    selected_model_name = "Logistic Regression" 
    classifier = LogisticRegression(random_state=42, solver='liblinear', C=1.0)
    
    # Jika Anda ingin meniru "Model DecisionTreeClassifier" dari gambar, Anda bisa ganti:
    # from sklearn.tree import DecisionTreeClassifier
    # selected_model_name = "Decision Tree Classifier"
    # classifier = DecisionTreeClassifier(random_state=42)
    # --------------------------------------------------------------

    text_processor = TfidfVectorizer(max_features=3000, ngram_range=(1,2))
    model_pipeline = Pipeline([
        ('tfidf', text_processor),
        ('classifier', classifier) 
    ])

    st.subheader("⚙️ Pelatihan Model")
    with st.spinner(f"Melatih model {selected_model_name}..."):
        try:
            model_pipeline.fit(X_train, y_train)
            st.session_state.sentiment_model_pipeline = model_pipeline
            st.session_state.X_test_sentiment = X_test 
            st.session_state.y_test_sentiment = y_test
            st.session_state.model_name_trained = selected_model_name # Simpan nama model
            st.success(f"Model {selected_model_name} berhasil dilatih!")
        except ValueError as e:
            st.error(f"Error saat melatih model: {e}")
            st.stop()


    if 'sentiment_model_pipeline' in st.session_state:
        st.subheader("📊 Hasil Evaluasi Model pada Data Uji")
        
        model_trained = st.session_state.sentiment_model_pipeline
        X_test_eval = st.session_state.X_test_sentiment
        y_test_eval = st.session_state.y_test_sentiment
        model_name_display = st.session_state.get('model_name_trained', "Classifier") # Ambil nama model

        # Menampilkan nama model dengan gaya seperti di gambar
        st.markdown(f"""
        <div style="background-color:#e6ffe6; padding: 10px; border-radius: 5px; margin-bottom: 15px; text-align: center;">
            <h4 style="color: #006400; margin:0;">Model: {model_name_display}</h4>
        </div>
        """, unsafe_allow_html=True)

        try:
            y_pred = model_trained.predict(X_test_eval)
            
            # Hitung metrik individual
            accuracy = accuracy_score(y_test_eval, y_pred)
            # Untuk presisi, recall, f1-score, kita gunakan rata-rata 'weighted' untuk menangani imbalanced classes
            # atau 'macro' jika Anda ingin setiap kelas berkontribusi sama.
            # 'binary' jika hanya ada 1 kelas positif (tidak relevan di sini karena ada 0 dan 1)
            # Jika Anda ingin per kelas, Anda perlu mengambilnya dari classification_report
            precision = precision_score(y_test_eval, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test_eval, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test_eval, y_pred, average='weighted', zero_division=0)

            # Tampilkan metrik menggunakan st.columns dan st.metric
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Accuracy", value=f"{accuracy*100:.1f}%")
                st.metric(label="Recall", value=f"{recall*100:.1f}%")
            with col2:
                st.metric(label="Precision", value=f"{precision*100:.1f}%")
                st.metric(label="F1-Score", value=f"{f1*100:.1f}%")

            st.markdown("---") # Garis pemisah

            st.text("Laporan Klasifikasi Lengkap:")
            target_names = ['Negatif (0)', 'Positif (1)'] 
            report_dict = classification_report(y_test_eval, y_pred, target_names=target_names, output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(report_dict).transpose())

            st.text("Matriks Konfusi:")
            cm = confusion_matrix(y_test_eval, y_pred)
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names, ax=ax_cm)
            ax_cm.set_xlabel("Prediksi")
            ax_cm.set_ylabel("Aktual")
            st.pyplot(fig_cm)

        except Exception as e:
            st.error(f"Error saat evaluasi model: {e}")

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
                cleaned_comment = simple_text_cleaner(user_comment_input)
                
                if not cleaned_comment.strip():
                    st.error("Komentar menjadi kosong setelah pembersihan dasar. Tidak dapat diprediksi.")
                else:
                    try:
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
    Dataset yang digunakan dalam aplikasi ini di ambil dari Kaggle.
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
    2.  **Vektorisasi Teks**: `cleaned_content` diubah menjadi fitur numerik menggunakan TF-IDF (Term Frequency-Inverse Document Frequency).
    3.  **Model Klasifikasi**: Model seperti `Logistic Regression` (atau model lain yang dipilih) dilatih pada fitur TF-IDF.
    4.  **Evaluasi**: Kinerja model diukur menggunakan Akurasi, Presisi, Recall, F1-Score, dan Matriks Konfusi.
    """)

    st.header("Cara Menggunakan Aplikasi")
    st.write("""
    1.  **Navigasi**: Gunakan menu dropdown di sidebar kiri untuk berpindah antar halaman.
    2.  **Halaman Awal**: Menampilkan sampel data, distribusi sentimen, dan visualisasi fitur.
    3.  **Model Sentimen**: Melatih model dan menampilkan kinerjanya dalam format kartu (Akurasi, Presisi, Recall, F1-Score), laporan klasifikasi, dan matriks konfusi.
    4.  **Prediksi Sentimen**: Memasukkan teks baru untuk diprediksi sentimennya.
    5.  **Informasi Aplikasi**: Penjelasan mengenai aplikasi.
    """)

    st.header("Tentang Proyek Ini")
    st.write("""
    Aplikasi ini adalah contoh bagaimana teknik pemrosesan bahasa alami (NLP) dan machine learning 
    dapat diterapkan untuk analisis sentimen.
    """)
    st.markdown("---")
    st.caption("Versi Aplikasi: 2.1.0 (Sentiment Analysis with Metric Cards)")
