import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# --- Konfigurasi Halaman ---
st.set_page_config(
    layout="wide",
    page_title="Personality Classification App",
    page_icon="🧠"
)

# --- Load Data & Model ---
@st.cache_data
def load_data():
    try:
        # Sesuaikan path dataset jika perlu
        df = pd.read_csv('dataset/personality_dataset.csv')
        return df
    except FileNotFoundError:
        return None

@st.cache_resource
def load_model():
    try:
        with open('new_personality_model.pkl', 'rb') as f:
            artifact = pickle.load(f)
        return artifact
    except FileNotFoundError:
        return None

df = load_data()
artifact = load_model()

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
# st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3074/3074767.png", width=100)

# --- PERUBAHAN 2: Mengganti radio dengan selectbox (Dropdown) ---
nav = st.sidebar.selectbox(
    "Go to", 
    ("Home", "Dataset", "EDA (Exploratory Data Analysis)", "Modelling", "Classification", "About")
)

# --- 1. HOME ---
if nav == "Home":
    st.title("🧠 Personality Classification App")
    st.subheader("Extrovert vs. Introvert Behavior Analysis")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://img.freepik.com/free-vector/introvert-extrovert-concept-illustration_114360-6362.jpg?w=740", caption="Personality Types")
    with col2:
        st.write("""
        ### Latar Belakang
        Aplikasi ini dikembangkan untuk mengklasifikasikan kepribadian seseorang menjadi **Extrovert** atau **Introvert** berdasarkan pola perilaku sosial mereka.
        
        **Fitur yang dianalisis meliputi:**
        * ⏱️ **Time Spent Alone**: Waktu yang dihabiskan sendirian.
        * 🎤 **Stage Fear**: Ketakutan berbicara di depan umum.
        * 🎉 **Social Event Attendance**: Frekuensi menghadiri acara sosial.
        * 🌳 **Going Outside**: Seberapa sering keluar rumah.
        * 🔋 **Drained after Socializing**: Perasaan lelah setelah bersosialisasi.
        * 👥 **Friends Circle Size**: Jumlah teman dalam lingkaran pertemanan.
        * 📱 **Post Frequency**: Frekuensi posting di media sosial.
        """)
        
        st.info("Gunakan menu di sebelah kiri untuk mengeksplorasi data, melihat performa model, atau melakukan klasifikasi langsung!")


# --- 2. DATASET ---
elif nav == "Dataset":
    # st.header("📂 Dataset Overview")
    # if df is not None:
    #     st.dataframe(df.head(10))
    #     st.write(f"Total Data: {df.shape[0]} Baris, {df.shape[1]} Kolom")
    #     st.write("Statistik Deskriptif:")
    #     st.dataframe(df.describe())
    # else:
    #     st.error("Dataset tidak ditemukan. Pastikan file 'personality_dataset.csv' ada di folder dataset.")
    st.header("📂 Dataset Overview")
    
    if df is not None:
        # Kita bagi menjadi 3 Tahap
        tab1, tab2, tab3 = st.tabs(["📄 Data Asli", "✨ Data Imputasi (Clean)", "⚙️ Data Siap Model"])
        
        # --- TAB 1: DATA ASLI ---
        with tab1:
            st.subheader("1. Data Mentah (Raw Data)")
            st.write("Data ini dibaca langsung dari file CSV tanpa perubahan apa pun.")
            st.dataframe(df.head(10))
            st.caption(f"Dimensi: {df.shape[0]} Baris, {df.shape[1]} Kolom")
            
            # Cek Missing Value
            null_counts = df.isnull().sum()
            if null_counts.sum() > 0:
                st.warning(f"⚠️ Ditemukan {null_counts.sum()} data kosong (missing values) di dataset ini.")
            else:
                st.success("✅ Tidak ditemukan data kosong pada dataset ini (Clean).")

        # --- TAB 2: DATA IMPUTASI ---
        with tab2:
            st.subheader("2. Data Hasil Imputasi (Handling Missing Values)")
            st.write("""
            Tahap ini mengisi data yang kosong (NaN). 
            - Kolom **Numerik** diisi dengan nilai **Median**.
            - Kolom **Kategori** diisi dengan nilai **Modus** (paling sering muncul).
            """)
            
            if artifact is not None:
                try:
                    # Ambil object preprocessors
                    preprocessors = artifact['preprocessors']
                    imputer_num = preprocessors['imputer_num']
                    imputer_cat = preprocessors['imputer_cat']
                    num_cols = preprocessors['numeric_cols']
                    cat_cols = preprocessors['categorical_cols']
                    
                    # Buat copy dataframe biar aman
                    df_imputed = df.drop('Personality', axis=1, errors='ignore').copy()
                    
                    # PROSES IMPUTASI
                    # 1. Imputasi Numerik
                    df_imputed[num_cols] = imputer_num.transform(df_imputed[num_cols])
                    # 2. Imputasi Kategori
                    df_imputed[cat_cols] = imputer_cat.transform(df_imputed[cat_cols])
                    
                    # Tampilkan
                    st.dataframe(df_imputed.head(10))
                    st.info("Data di atas sudah bersih dari nilai kosong, namun formatnya masih asli (belum di-encode).")
                    
                except Exception as e:
                    st.error(f"Gagal memuat data imputasi: {e}")
            else:
                st.warning("⚠️ Model belum dimuat. Jalankan `models.py` terlebih dahulu.")

        # --- TAB 3: DATA PRE-PROCESSED ---
        with tab3:
            st.subheader("3. Data Siap Model (Encoded)")
            st.write("Data ini adalah hasil akhir yang akan masuk ke algoritma Machine Learning.")
            st.markdown("""
            * **Encoding:** Mengubah teks (Yes/No) menjadi angka (1/0).
            """)
            
            if artifact is not None:
                try:
                    
                    encoders = preprocessors['encoders']
                    # scaler = preprocessors['scaler']
                    
                    # Gunakan data hasil imputasi tadi
                    df_ready = df_imputed.copy()
                    
                    # PROSES ENCODING
                    for col in cat_cols:
                        if col in df_ready.columns:
                            le = encoders[col]
                            df_ready[col] = le.transform(df_ready[col])
                    
                    # # PROSES SCALING
                    # df_ready[num_cols] = scaler.transform(df_ready[num_cols])
                    
                    # Tampilkan
                    st.dataframe(df_ready.head(10))
                    st.caption("Perhatikan semua nilai kini berupa angka desimal (float) atau integer.")
                    
                    st.markdown("#### Statistik Data Final")
                    st.dataframe(df_ready.describe())
                    
                except Exception as e:
                    st.error(f"Gagal memproses data final: {e}")
            else:
                st.warning("Model belum dimuat.")
    else:
        st.error("File 'personality_datasert.csv' tidak ditemukan.")

# --- 3. EDA ---
elif nav == "EDA (Exploratory Data Analysis)":
    st.header("📊 Exploratory Data Analysis")
    if df is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribusi Target (Personality)")
            fig, ax = plt.subplots()
            # Ganti 'Personality' dengan nama kolom target di csv Anda jika berbeda
            if 'Personality' in df.columns:
                sns.countplot(x='Personality', data=df, palette='viridis', ax=ax)
                plt.xticks(rotation=45)
                st.pyplot(fig)
            else:
                st.warning("Kolom 'Personality' tidak ditemukan.")

        with col2:
            st.subheader("Korelasi Fitur Numerik")
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                fig2, ax2 = plt.subplots()
                sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm', ax=ax2)
                st.pyplot(fig2)
            else:
                st.warning("Tidak ada data numerik untuk korelasi.")
    else:
        st.error("Data tidak tersedia.")

# --- 4. MODELLING (PERUBAHAN 3: DINAMIS) ---
elif nav == "Modelling":
    st.header("⚙️ Model Performance Evaluation")
    
    if artifact and 'history' in artifact:
        st.write("Berikut adalah perbandingan performa model yang dilakukan saat proses training:")
        
        # Ambil data history dari artifact
        history_data = artifact['history']
        df_compare = pd.DataFrame(history_data)
        
        # Tampilkan Tabel
        st.subheader("Tabel Akurasi")
        st.dataframe(
            df_compare.style.format({
                'Akurasi Baseline': '{:.2%}',
                'Akurasi Tuned': '{:.2%}',
                'Improvement': '{:+.2%}'
            })
        )

        # Kesimpulan Dinamis
        best_model_name = artifact['model_name']
        best_acc = artifact['accuracy']
        st.success(f"🏆 **Kesimpulan:** Model terbaik adalah **{best_model_name}** dengan akurasi **{best_acc:.2%}**.")
        
        # Visualisasi Grafik Batang
        st.subheader("Grafik Perbandingan Model")
        
        # Melt data untuk keperluan plotting seaborn
        df_melted = df_compare.melt(id_vars="Model", value_vars=["Akurasi Baseline", "Akurasi Tuned"], var_name="Tipe", value_name="Akurasi")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=df_melted, x="Model", y="Akurasi", hue="Tipe", palette="magma", ax=ax)
        ax.set_ylim(0, 1.1)
        
        # Menambah label angka di atas batang
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f')
            
        st.pyplot(fig)
        
        
    else:
        st.warning("⚠️ Data riwayat training tidak ditemukan di dalam model. Harap jalankan ulang `models.py` yang baru.")
        # Fallback statis jika user belum run models.py baru
        st.write("*(Menampilkan data statis sebagai placeholder)*")
        st.write("KNN Accuracy: 85% (Contoh)")

# --- 5. CLASSIFICATION ---
elif nav == "Classification":
    st.header("🔮 Klasifikasi Kepribadian")
    st.write("Jawablah pertanyaan di bawah ini dengan jujur untuk mendapatkan hasil yang akurat.")
    
    if artifact is None:
        st.error("Model belum dimuat. Jalankan `models.py` terlebih dahulu.")
    else:
        model = artifact['model']
        preprocessors = artifact['preprocessors']
        
        encoders = preprocessors['encoders']
        # scaler = preprocessors['scaler']
        num_cols = preprocessors['numeric_cols']
        cat_cols = preprocessors['categorical_cols']
        target_enc = preprocessors['target_encoder']
        
        # --- KAMUS PERTANYAAN (MAPPING) ---
        # Di sini kita mengubah nama kolom teknis menjadi pertanyaan bahasa Indonesia
        feature_questions = {
            "Time_spent_Alone": "Berapa jam rata-rata waktu yang Anda habiskan sendirian (per hari)?",
            "Stage_fear": "Apakah Anda merasa takut atau grogi saat berbicara di depan umum?",
            "Social_event_attendance": "Berapa kali Anda menghadiri acara sosial dalam seminggu?",
            "Going_outside": "Seberapa sering Anda pergi keluar rumah (untuk main/hangout)?",
            "Drained_after_socializing": "Apakah energi Anda merasa terkuras/lelah setelah bersosialisasi?",
            "Friends_circle_size": "Berapa kira-kira jumlah teman dalam lingkaran pertemanan Anda?",
            "Post_frequency": "Seberapa sering Anda membuat postingan di Media Sosial?"
        }

        with st.form("prediction_form"):
            user_input = {}
            
            # Kita bagi layout jadi 2 Kolom (Kiri & Kanan)
            c1, c2 = st.columns(2)
            
            # --- KOLOM KIRI ---
            with c1:
                
                # Input 1: Time_spent_Alone (Float)
                user_input['Time_spent_Alone'] = st.number_input(
                    "Berapa jam rata-rata waktu Anda sendirian (per hari)?", 
                    min_value=0.0, max_value=24.0, value=4.0
                )
                
                # Input 2: Social_event_attendance (Integer)
                user_input['Social_event_attendance'] = st.number_input(
                    "Berapa kali menghadiri acara sosial dalam seminggu?", 
                    min_value=0, step=1, value=1
                )
                
                # Input 3: Drained_after_socializing (Kategorikal - Selectbox)
                # Ambil opsi Yes/No dari encoder
                opsi_drained = list(encoders['Drained_after_socializing'].classes_)
                user_input['Drained_after_socializing'] = st.selectbox(
                    "Apakah energi terkuras setelah bersosialisasi?", 
                    opsi_drained
                )

                # Input 4: Post_frequency (Slider)
                user_input['Post_frequency'] = st.slider(
                    "Seberapa sering posting di Sosmed (Skala 0-10)?",
                    min_value=0, max_value=10, value=2
                )

            # --- KOLOM KANAN ---
            with c2:
                
                # Input 5: Stage_fear (Kategorikal - Selectbox)
                opsi_fear = list(encoders['Stage_fear'].classes_)
                user_input['Stage_fear'] = st.selectbox(
                    "Apakah takut/grogi bicara di depan umum?", 
                    opsi_fear
                )
                
                # Input 6: Going_outside (Slider)
                user_input['Going_outside'] = st.slider(
                    "Seberapa sering pergi keluar rumah/hangout (Skala 0-10)?",
                    min_value=0, max_value=10, value=5
                )
                
                # Input 7: Friends_circle_size (Integer)
                user_input['Friends_circle_size'] = st.number_input(
                    "Berapa jumlah teman dekat dalam circle Anda?", 
                    min_value=0, step=1, value=3
                )

            st.markdown("---")
            submit = st.form_submit_button("🔍 Analisis Sekarang")
        
        if submit:
            # Proses Prediksi (Sama seperti sebelumnya)
            input_df = pd.DataFrame([user_input])
            
            try:
                # Encode & Scale
                for col in cat_cols:
                    le = encoders[col]
                    input_df[col] = le.transform(input_df[col])
                
                # input_df[num_cols] = scaler.transform(input_df[num_cols])
                
                # Prediksi
                # Menyusun urutan kolom agar sesuai model
                expected_order = [
                    "Time_spent_Alone", 
                    "Stage_fear", 
                    "Social_event_attendance", 
                    "Going_outside", 
                    "Drained_after_socializing", 
                    "Friends_circle_size", 
                    "Post_frequency"
                ]

                input_final = input_final = input_df[expected_order]
                
                prediction_idx = model.predict(input_final)[0]
                prediction_label = target_enc.inverse_transform([prediction_idx])[0]
                
                st.success(f"### Hasil: Kepribadian Anda adalah **{prediction_label}**")
                # st.balloons()
                
            except Exception as e:
                st.error(f"Error: {e}")

# --- 6. ABOUT ---
elif nav == "About":
    st.header("Tentang Saya")
    c1, c2 = st.columns([1,2])

    with c1:
        st.image("photos/foto-profil.jpg", width=400)

    with c2:
        st.subheader("Ramadhan Renaldy, M.Kom")
        
        st.write("""Saat ini saya bekerja sebagai Dosen di Universitas PGRI Semarang. 
                Fokus pengajaran dan penelitian saya meliputi Dasar & Algoritma Pemrograman, Data Science, serta Business Intelligence.""")
                 
        st.write("""Saya telah menempuh pendidikan Sarjana dan Magister pada bidang Teknik Informatika.
                Sejalan dengan latar belakang akademis tersebut, penelitian saya sebelumnya juga menyangkut mengenai Data Science.""")
                 
        st.write("""Dalam keseharian, saya fokus mengajar mata kuliah:""")
        st.write("""💻 **Dasar & Algoritma Pemrograman**""")
        st.write("""📊 **Data Science**""")
        st.write("""📈 **Business Intelligence**""")
                
        st.write("""Aplikasi klasifikasi kepribadian ini dikembangkan sebagai bentuk implementasi keilmuan dalam menerapkan algoritma Machine Learning untuk menyelesaikan Uji Kompetensi BNSP Ilmuwan Data Muda.

                """)
