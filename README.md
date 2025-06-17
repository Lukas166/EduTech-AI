# EduTech AI Chatbot

**Website Pembelajaran Computer Science dengan Teknologi Artificial Intelligence**

EduTech adalah chatbot edukasi interaktif berbasis AI yang bertujuan mempermudah pembelajaran Teknik Informatika (TI) secara mandiri dan fleksibel. Proyek ini menggabungkan pendekatan RAG (Retrieval-Augmented Generation) serta layanan multibahasa untuk memberikan jawaban akurat dan kontekstual terhadap pertanyaan pengguna.

---

## Live Website & Repository

- 🌐 [Akses Website Chatbot](https://edutech-ai-chatbot.streamlit.app)
- 📂 [Repository GitHub](https://github.com/Lukas166/EduTech-AI)

---

## Mengapa EduTech?

> Dalam era saat ini, banyak orang membutuhkan bantuan untuk memahami konsep teoritis dalam bidang Teknik Informatika. Sayangnya, tidak semua orang memiliki akses yang mudah ke sumber belajar yang interaktif dan responsif, baik karena keterbatasan media pembelajaran, waktu, maupun bimbingan personal.

---

## Tujuan Proyek (3M)

- **Mempermudah** akses pembelajaran dengan memberikan fitur chatbot yang interaktif.  
- **Menjawab** pertanyaan seputar Teknik Informatika secara akurat dan relevan.  
- **Memandu** pengguna ke materi pembelajaran lanjutan yang tersedia di dalam maupun luar aplikasi.

---

## Solusi yang Kami Tawarkan

Chatbot edukasi berbasis AI yang:
- Memahami konteks pertanyaan pengguna.
- Menggunakan dataset pertanyaan dan sistem RAG untuk memberikan jawaban tepat.
- Mendukung interaksi dalam berbagai bahasa.

---

## Fitur Utama

- Chatbot interaktif & multi-bahasa  
- Respons kontekstual menggunakan NLP  
- Rekomendasi materi belajar  
- Lebih dari 100 materi pembelajaran TI  
- Dapat digunakan sebagai tutor mandiri

---

## Alur Kerja Sistem

```
FAQ Dataset → Embedding (allminiLM) → FAQ dataset embedded  
User Input → Google Translate → Embedding → Semantic Search (FAQ dataset embedded) → Gemini LLM → Chatbot Answer
```

---

## Metode AI yang Digunakan

- **Fine-Tuning**: Melatih ulang model AllMiniLM dengan dataset pertanyaan TI untuk akurasi tinggi.  
- **Embedding**: Mengubah teks menjadi vektor untuk keperluan Semantic Search.  
- **RAG (Retrieval-Augmented Generation)**: Mengkombinasikan retrieval dan generasi jawaban dengan konteks dataset.

---

## Tools & Teknologi

- Google Colab
- Sentence-BERT (SBERT)
- Hugging Face Transformers
- Gemini API
- Google Translate API
- Python, JSON
- Streamlit

---

## Instalasi untuk menjalankan di Lokal

1. **Clone repository:**
   ```bash
   git clone https://github.com/Lukas166/EduTech-AI.git
   cd EduTech-AI
   ```

2. **Buat file `.env`:**
   ```env
   GEMINI_API_KEY="masukkan_API_KEY_anda_disini"
   ```

3. **Install dependensi:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan aplikasi:**
   ```bash
   streamlit run app.py
   ```
   Aplikasi akan berjalan di `http://localhost:8501`

---

## 📁 Struktur Folder (Contoh)

```
EduTech-AI/
├── app.py
├── .env.example
├── requirements.txt
├── data/
│   ├── faq_dataset.csv
│   ├── embedded_dataset.pkl
│   └── ...
├── model/
│   └── fine_tuned_model/
├── utils/
│   └── helper_functions.py
└── assets/
    └── ui_images/
```

---

## Keterangan Aplikasi

Website kami berfokus pada tampilan edukatif yang menampilkan kursus terkait *Computer Science* dengan bantuan chatbot interaktif. Berikut langkah-langkah menggunakan aplikasi:

### 1. **Dashboard (Halaman Awal)**  
- Saat pengguna membuka website, akan langsung diarahkan ke halaman **Dashboard**.  
- Di halaman ini, pengguna bisa melihat:
  - Penjelasan singkat tentang AI Chatbot yang tersedia.
  - Daftar beberapa course utama yang telah tersedia.
- **Sidebar** akan menampilkan tiga menu utama:
  - `Dashboard`
  - `Chatbot`
  - `Course List`

### 2. **AI Learning Assistant (Chatbot)**
- Fitur chatbot dapat diakses melalui tab `Chatbot` di sidebar.
- Chatbot ini bersifat **multi-language** dan mampu memberikan **respons interaktif dan kontekstual**.
- Setelah pengguna memberikan pertanyaan (dalam berbagai bahasa), sistem akan:
  - Mendeteksi bahasa secara otomatis.
  - Memberikan jawaban yang relevan dalam bahasa yang sama.
  - Menyertakan **rekomendasi link** ke halaman *course* yang terkait.

### 3. **Navigasi ke Course**
- Jika pengguna mengklik tautan yang disarankan oleh chatbot, mereka akan diarahkan langsung ke halaman *course* yang dimaksud.
- Hal ini membantu pembelajaran secara langsung dan mendalam sesuai konteks pertanyaan yang diajukan.

### 4. **Course List (Daftar Kursus)**
- Tab terakhir di sidebar adalah `Course List`, yang menampilkan semua materi kursus yang tersedia.
- Pengguna dapat menjelajahi seluruh konten pembelajaran secara mandiri dan memilih topik yang mereka minati.