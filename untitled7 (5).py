import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
import numpy as np
import csv
import re

# Konfigurasi halaman
st.set_page_config(page_title="Sentiment Analysis App", layout="wide")

# Fungsi untuk membersihkan teks dasar
def simple_text_cleaner(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text.strip()

# Fungsi untuk memuat dan memproses data
@st.cache_data
def load_data(url):
    try:
        df = pd.read_csv(url, engine='python', quoting=csv.QUOTE_MINIMAL, on_bad_lines='skip')
    except pd.errors.ParserError:
        st.error("Gagal membaca CSV. Mencoba parameter berbeda...")
        try:
            df = pd.read_csv(url, engine='python', quoting=csv.QUOTE_NONE, escapechar='\\', on_bad_lines='skip')
        except Exception as e:
            st.error(f"Gagal membaca CSV: {e}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Kesalahan memuat data: {e}")
        return pd.DataFrame()

    if 'content' in df.columns:
       df['content'] = df['content'].astype(str).fillna('')
       df['cleaned_content'] = df['content'].apply(simple_text_cleaner)
       df['panjang karakter'] = df['cleaned_content'].str.len()
       df['jumlah kata'] = df['cleaned_content'].str.split().str.len()


    else:
        st.warning("Kolom 'content' tidak ditemukan.")
        df['cleaned_content'] = ""
        df['panjang karakter'] = 0
        df['jumlah kata'] = 0

    if 'score' not in df.columns:
        st.error("Kolom 'score' tidak ditemukan. Tidak dapat membuat target sentimen.")
        df['Sentiment'] = 'Unknown'
        df['Sentiment_Label'] = -1 # Label default untuk unknown
    else:
        df['score'] = pd.to_numeric(df['score'], errors='coerce')
        df.dropna(subset=['score'], inplace=True)
        
        # Mendefinisikan sentimen: Negatif (0), Netral (1), Positif (2)
        conditions = [
            (df['score'] <= 2),  # Skor 1, 2 -> Negatif
            (df['score'] == 3),  # Skor 3 -> Netral
            (df['score'] >= 4)   # Skor 4, 5 -> Positif
        ]
        choices_text = ['Negatif', 'Netral', 'Positif']
        choices_label = [0, 1, 2] # Negatif: 0, Netral: 1, Positif: 2

        df['Sentiment'] = np.select(conditions, choices_text, default='Unknown')
        df['Sentiment_Label'] = np.select(conditions, choices_label, default=-1) # Assign -1 untuk yang tidak masuk kategori

        # Filter out 'Unknown' sentiments if any were not caught by conditions
        df = df[df['Sentiment'] != 'Unknown']
        
        if df.empty:
            st.error("Tidak ada data valid untuk sentimen setelah pemrosesan skor.")
            return pd.DataFrame()

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

numerical_cols_original = ['panjang karakter', 'jumlah kata']

numerical_cols_for_viz = [col for col in numerical_cols_original if col in df_processed.columns]


# Sidebar menu
menu_options = ["Halaman Awal", "Model", "Prediksi", "Informasi Aplikasi", "Anggota Kelompok" ]
menu = st.sidebar.selectbox("📁 Navigasi", menu_options)

# ====================== HALAMAN AWAL ======================
if menu == "Halaman Awal":
    st.title("📊 Dashboard Data Ulasan Pengguna")

    st.subheader("Dataset Asli (Mentah)")
    if not df_original_raw.empty:
        st.markdown(f"Menampilkan **{len(df_original_raw)}** baris data asli.")
        # PERUBAHAN 1: Menampilkan seluruh df_original_raw
        st.dataframe(df_original_raw) 
    else:
        st.warning("Gagal menampilkan dataset asli mentah.")

    st.subheader("Dataset Setelah Pemrosesan dan Pembuatan Fitur")
    if not df_processed.empty:
        st.markdown(f"Menampilkan **{len(df_processed)}** baris data yang telah diproses.")
        st.dataframe(df_processed[['content', 'score', 'cleaned_content', 'panjang karakter', 'jumlah kata', 'Sentiment', 'Sentiment_Label']].head())
    else:
        st.warning("Gagal menampilkan dataset yang diproses.")

    st.subheader("Distribusi Sentimen")
    if 'Sentiment' in df_processed.columns and not df_processed.empty:
        fig_sentiment, ax_sentiment = plt.subplots()
        # PERUBAHAN 2.1: Menambahkan Netral ke palette
        palette_sentiment = {'Positif': 'green', 'Negatif': 'red', 'Netral': 'blue'}
        sns.countplot(x='Sentiment', data=df_processed, ax=ax_sentiment, 
                      order=['Negatif', 'Netral', 'Positif'], # Tentukan urutan jika diinginkan
                      palette=palette_sentiment)
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
            sns.histplot(data=df_processed, x=selected_col_hist, hue='Sentiment' if 'Sentiment' in df_processed.columns else None, kde=True, bins=30, ax=ax, hue_order=['Negatif', 'Netral', 'Positif'])
            ax.set_title(f'Distribusi {selected_col_hist}')
            st.pyplot(fig)

        elif viz_type == "Boxplot":
            selected_col_box = st.selectbox("Pilih fitur", numerical_cols_for_viz, key="box_select_home")
            fig, ax = plt.subplots()
            sns.boxplot(data=df_processed, x='Sentiment' if 'Sentiment' in df_processed.columns else None, y=selected_col_box, ax=ax, order=['Negatif', 'Netral', 'Positif'])
            ax.set_title(f'Boxplot {selected_col_box} berdasarkan Sentimen')
            st.pyplot(fig)
    else:
        st.info("Tidak cukup fitur numerik untuk divisualisasikan.")


# ====================== HALAMAN MODEL ======================
elif menu == "Model":
    st.title("🤖 Model Performance: Klasifikasi Sentimen (Multi-class)")

    if df_processed.empty or 'cleaned_content' not in df_processed.columns or 'Sentiment_Label' not in df_processed.columns:
        st.warning("Data yang diproses tidak lengkap atau tidak ada. Tidak dapat melatih model.")
        st.stop()
    
    df_model_data = df_processed.dropna(subset=['cleaned_content', 'Sentiment_Label'])
    df_model_data = df_model_data[df_model_data['cleaned_content'].str.strip() != '']
    df_model_data = df_model_data[df_model_data['Sentiment_Label'] != -1] # Pastikan hanya label valid

    if df_model_data.empty:
        st.error("Tidak ada data yang valid untuk melatih model setelah filter tambahan.")
        st.stop()

    X = df_model_data['cleaned_content']
    y = df_model_data['Sentiment_Label']

    # PERUBAHAN 2.2: Cek minimal 3 kelas untuk stratify jika multiclass
    min_samples_per_class_for_stratify = 2 # atau lebih tinggi jika diperlukan oleh stratify
    can_stratify = len(y.unique()) >= 3 and y.value_counts().min() >= min_samples_per_class_for_stratify

    if len(X) < 3 or len(y.unique()) < 3 : # Butuh setidaknya 3 sampel dan 3 kelas untuk multiclass yang berarti
        st.error(f"Tidak cukup data atau variasi kelas untuk melatih model multi-class. Jumlah data: {len(X)}, Jumlah kelas unik: {len(y.unique())}")
        st.stop()
        
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y if can_stratify else None
        )
    except ValueError as e:
        st.warning(f"Tidak bisa melakukan stratify: {e}. Melanjutkan tanpa stratify.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    selected_model_name = "Logistic Regression" 
    # Untuk multiclass, LogisticRegression default ke 'ovr' (one-vs-rest)
    classifier = LogisticRegression(random_state=42, solver='liblinear', C=1.0, multi_class='ovr') 
    
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
            st.session_state.model_name_trained = selected_model_name
            st.success(f"Model {selected_model_name} berhasil dilatih!")
        except ValueError as e:
            st.error(f"Error saat melatih model: {e}")
            st.stop()

    if 'sentiment_model_pipeline' in st.session_state:
        st.subheader("📊 Hasil Evaluasi Model pada Data Uji")
        
        model_trained = st.session_state.sentiment_model_pipeline
        X_test_eval = st.session_state.X_test_sentiment
        y_test_eval = st.session_state.y_test_sentiment
        model_name_display = st.session_state.get('model_name_trained', "Classifier")

        st.markdown(f"""
        <div style="background-color:#e6ffe6; padding: 10px; border-radius: 5px; margin-bottom: 15px; text-align: center;">
            <h4 style="color: #006400; margin:0;">Model: {model_name_display}</h4>
        </div>
        """, unsafe_allow_html=True)

        try:
            y_pred = model_trained.predict(X_test_eval)
            
            accuracy = accuracy_score(y_test_eval, y_pred)
            # 'weighted' average masih cocok untuk multiclass
            precision = precision_score(y_test_eval, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test_eval, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test_eval, y_pred, average='weighted', zero_division=0)

            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Accuracy", value=f"{accuracy*100:.1f}%")
                st.metric(label="Recall (Weighted Avg)", value=f"{recall*100:.1f}%")
            with col2:
                st.metric(label="Precision (Weighted Avg)", value=f"{precision*100:.1f}%")
                st.metric(label="F1-Score (Weighted Avg)", value=f"{f1*100:.1f}%")

            st.markdown("---") 

            st.text("Laporan Klasifikasi Lengkap:")
            # PERUBAHAN 2.3: Target names untuk multiclass
            target_names = ['Negatif (0)', 'Netral (1)', 'Positif (2)'] 
            report_dict = classification_report(y_test_eval, y_pred, target_names=target_names, output_dict=True, zero_division=0, labels=[0,1,2])
            st.dataframe(pd.DataFrame(report_dict).transpose())

            st.text("Matriks Konfusi:")
            cm = confusion_matrix(y_test_eval, y_pred, labels=[0,1,2]) # Pastikan urutan label
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=target_names, yticklabels=target_names, ax=ax_cm)
            ax_cm.set_xlabel("Prediksi")
            ax_cm.set_ylabel("Aktual")
            st.pyplot(fig_cm)

        except Exception as e:
            st.error(f"Error saat evaluasi model: {e}")

# ====================== HALAMAN PREDIKSI ======================
elif menu == "Prediksi":
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

                        # PERUBAHAN 2.4: Map sentimen untuk multiclass
                        sentiment_map = {2: 'Positif', 0: 'Negatif', 1: 'Netral'}
                        predicted_sentiment_text = sentiment_map.get(prediction_label, "Tidak diketahui")

                        st.subheader("Hasil Prediksi:")
                        if predicted_sentiment_text == 'Positif':
                            st.success(f"Sentimen: **{predicted_sentiment_text}**")
                        elif predicted_sentiment_text == 'Negatif':
                            st.error(f"Sentimen: **{predicted_sentiment_text}**")
                        elif predicted_sentiment_text == 'Netral':
                            st.info(f"Sentimen: **{predicted_sentiment_text}**") # Warna biru untuk netral
                        else:
                            st.warning(f"Sentimen: **{predicted_sentiment_text}**")
                        
                        st.write("Probabilitas:")
                        # Pastikan urutan probabilitas sesuai dengan kelas 0, 1, 2
                        # Urutan probabilitas dari predict_proba() biasanya sesuai dengan model.classes_
                        # Untuk LogisticRegression, ini adalah urutan numerik dari label.
                        # Jika model.classes_ adalah [0, 1, 2] maka proba[0] adalah Negatif, proba[1] Netral, proba[2] Positif
                        classes_order = model_to_predict.classes_ # Seharusnya [0, 1, 2]
                        
                        st.write(f"- Negatif (skor 1-2): {prediction_proba[np.where(classes_order == 0)[0][0]]:.2%}")
                        st.write(f"- Netral (skor 3): {prediction_proba[np.where(classes_order == 1)[0][0]]:.2%}")
                        st.write(f"- Positif (skor 4-5): {prediction_proba[np.where(classes_order == 2)[0][0]]:.2%}")
                        
                        st.markdown("---")
                        st.write("Komentar yang Dianalisis (setelah pembersihan dasar):")
                        st.text(cleaned_comment)

                    except IndexError:
                         st.error("Error dalam mengakses probabilitas. Periksa urutan kelas model.")
                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat prediksi: {e}")

# ====================== HALAMAN INFORMASI APLIKASI ======================
elif menu == "Informasi Aplikasi":
    st.title("ℹ️ Informasi Aplikasi Analisis Sentimen")
    st.markdown("---")

    st.header("Tujuan Aplikasi")
    st.write("""
    Aplikasi ini dibangun untuk melakukan analisis sentimen pada dataset ulasan pengguna. 
    Tujuannya adalah untuk mengklasifikasikan ulasan sebagai 'Positif', 'Negatif', atau 'Netral' 
    berdasarkan konten teksnya. Ini dapat membantu memahami opini publik atau feedback pelanggan secara lebih detail.
    Aplikasi ini juga dibangun dengan tujuan untuk menyelesaikan proyek mata kuliah Data Mining.
    """)

    st.header("Data yang Digunakan")
    st.write(f"""
    Dataset yang digunakan dalam aplikasi ini di ambil dari Kaggle.
    Dataset ini berisi ulasan pengguna, termasuk skor yang diberikan dan konten ulasan.
    Sentimen ('Positif'/'Negatif'/'Netral') ditentukan berdasarkan kolom 'score':
    - Skor 1-2: Negatif (Label 0)
    - Skor 3: Netral (Label 1)
    - Skor 4-5: Positif (Label 2)
    """)
    st.write("""
    Fitur tambahan yang dibuat dari data asli adalah:
    - **`cleaned_content`**: Konten ulasan setelah pembersihan dasar.
    - **`panjang karakter`**: Panjang karakter dari `cleaned_content`.
    - **`jumlah kata`**: Jumlah kata dalam `cleaned_content`.
    Fitur utama untuk model sentimen adalah representasi TF-IDF dari `cleaned_content`.
    """)

    st.header("Metode Analisis Sentimen")
    st.write("""
    Metode yang digunakan adalah **Klasifikasi Teks supervised learning (multi-class)**.
    1.  **Preprocessing Teks**: Teks ulasan dibersihkan.
    2.  **Vektorisasi Teks**: `cleaned_content` diubah menjadi fitur numerik menggunakan TF-IDF.
    3.  **Model Klasifikasi**: Model `Logistic Regression` (dikonfigurasi untuk multi-class) dilatih pada fitur TF-IDF.
    4.  **Evaluasi**: Kinerja model diukur menggunakan Akurasi, Presisi (rata-rata tertimbang), Recall (rata-rata tertimbang), F1-Score (rata-rata tertimbang), dan Matriks Konfusi untuk ketiga kelas.
    """)

    st.header("Cara Menggunakan Aplikasi")
    st.write("""
    1.  **Navigasi**: Gunakan menu dropdown di sidebar kiri.
    2.  **Halaman Awal**: Menampilkan seluruh dataset mentah, sampel data yang diproses, distribusi sentimen (termasuk Netral), dan visualisasi fitur.
    3.  **Model Sentimen**: Melatih model multi-class dan menampilkan kinerjanya.
    4.  **Prediksi Sentimen**: Memasukkan teks baru untuk diprediksi sentimennya (Positif, Negatif, atau Netral).
  

    """)
    st.markdown("---")
    st.caption("Versi Aplikasi: 1.1.0")

elif menu == "Anggota Kelompok":
   
    st.header("Anggota Kelompok 9")
    st.markdown("---")
    st.write("""
    - IDA AYU PRADIPTA NARASWARI YONI (2304030049).
    - AGUSTINUS JUAN JOSEPH ABANAT (2304030051).
    - ERA FEBI SULISTIAWATI (2304030076).
    - SHIERA NABILA FIRNANDA  (2304030077).
    """)
