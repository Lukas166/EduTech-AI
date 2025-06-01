import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from googletrans import Translator, LANGUAGES
from langdetect import detect
import os
from dotenv import load_dotenv

load_dotenv()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Menyembunyikan pesan INFO dan WARNING TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

gemini_api_key = os.getenv("GEMINI_API_KEY")

# Konfigurasi Gemini dengan parameter creativity
genai.configure(api_key=gemini_api_key)

# Fungsi bantu untuk preprocessing teks
def preprocess_text(text):
    return ' '.join(text.lower().strip().split())

class RAGChatbot:
    def __init__(self, faq_file="faq.json", model_path="best_embedding_model", top_k=5, max_history=10, 
                 temperature=1, top_p=0.9, top_k_gen=40):
        self.top_k = top_k
        self.max_history = max_history
        self.chat_history = []
        
        # Parameter untuk Gemini creativity
        self.temperature = temperature
        self.top_p = top_p
        self.top_k_gen = top_k_gen
        
        # Inisialisasi Gemini model dengan generation config
        self.gemini_model = genai.GenerativeModel(
            "gemini-2.0-flash",
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k_gen,
                max_output_tokens=2048,
            )
        )
        
        # Inisialisasi translator
        self.translator = Translator()
        
        self.system_prompt = """You are a helpful CS (Computer Science) assistant bot. Your role is to help answer questions related to computer science concepts based ONLY on the provided context.

RULES:
1. Answer in english by default.
2. Answer ONLY based on the provided context from the knowledge base (you must compare user input with given responses too.
3. You can respond to greetings (hi, hello, thanks, goodbye) in a friendly way, you can also answer about who you are.
4. For CS questions NOT covered in the context, politely say you cant recommend the topic from the course, but explain it a bit and recommend them study outside, give the reference.
6. If the similarity of context quite low, check correctly if there is a misstyping, and ask again. example: user input: abstrction, answer: do you mean abstraction, if it is, then bla bla...
7. If the context doesn't contain relevant information for the question, admit you don't know, and ask the question again.
8. Please explain it further in new paragraph after, relevant to the answer.
9. If user ask you to tell them more about the relevant context, you're allow to do it with your AI model.
"""

        # Load FAQ embeddings dan model embedding
        self.faq_data = self._load_json(faq_file)
        self.embedding_model = SentenceTransformer(model_path)
        self.embeddings_matrix = np.array([entry['embedding'] for entry in self.faq_data])

    def _load_json(self, file_path):
        """Load data JSON dan tangani error jika file tidak ditemukan"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"Loaded {len(data)} entries from {file_path}")
                return data
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"Error loading {file_path}.")
            return []

    def detect_language(self, text):
        """Deteksi bahasa dari input text"""
        try:
            detected_lang = detect(text)
            print(f"Detected language: {detected_lang}")
            return detected_lang
        except:
            print("Language detection failed, assuming English")
            return 'en'

    def translate_to_english(self, text, source_lang):
        """Translate text ke bahasa Inggris jika bukan bahasa Inggris"""
        if source_lang == 'en':
            return text
        
        try:
            translated = self.translator.translate(text, src=source_lang, dest='en')
            print(f"Translated to English: {translated.text}")
            return translated.text
        except Exception as e:
            print(f"Translation error: {e}")
            return text

    def translate_from_english(self, text, target_lang):
        """Translate response dari bahasa Inggris ke bahasa target"""
        if target_lang == 'en':
            return text
            
        try:
            translated = self.translator.translate(text, src='en', dest=target_lang)
            print(f"Translated to {target_lang}: {translated.text}")
            return translated.text
        except Exception as e:
            print(f"Translation error: {e}")
            return text

    def _search_context(self, query):
        """
        Mencari context relevan untuk query user menggunakan cosine similarity.
        Mengembalikan list context dengan skor similarity.
        """
        query_embedding = self.embedding_model.encode([preprocess_text(query)])
        similarities = cosine_similarity(query_embedding, self.embeddings_matrix)[0]
        top_indices = np.argsort(similarities)[-self.top_k:][::-1]

        return [{
            'tag': self.faq_data[idx]['tag'],
            'pattern': self.faq_data[idx]['original_pattern'],
            'responses': self.faq_data[idx]['responses'],
            'similarity': similarities[idx]
        } for idx in top_indices]

    def _format_context(self, contexts):
        """Format context yang relevan untuk dimasukkan ke prompt Gemini"""
        if not contexts:
            return "No relevant context found."

        lines = ["KNOWLEDGE BASE CONTEXT:"]
        for i, ctx in enumerate(contexts, 1):
            lines.append(
                f"\n{i}. Topic: {ctx['tag']}\n"
                f"   Question Pattern: {ctx['pattern']}\n"
                f"   Responses: {'; '.join(ctx['responses'])}\n"
                f"   Relevance Score: {ctx['similarity']:.4f}"
            )
        return '\n'.join(lines)

    def _format_history(self):
        """Format chat history untuk konteks prompt"""
        if not self.chat_history:
            return ""
        return "\nCHAT HISTORY:\n" + '\n\n'.join(
            f"User: {h['user']}\nAssistant: {h['assistant']}"
            for h in self.chat_history[-self.max_history:]
        )

    def _update_history(self, user, assistant):
        """Tambahkan percakapan ke history dan batasi ke max_history"""
        self.chat_history.append({'user': user, 'assistant': assistant})
        if len(self.chat_history) > self.max_history:
            self.chat_history = self.chat_history[-self.max_history:]

    def generate_response(self, user_query):
        """
        Fungsi utama untuk menghasilkan respon dengan multilingual support:
        1. Deteksi bahasa input
        2. Translate ke English jika perlu
        3. Semantic search
        4. Generate response dengan Gemini
        5. Translate response kembali ke bahasa input
        """
        
        # Step 1: Deteksi bahasa input
        detected_lang = self.detect_language(user_query)
        
        # Step 2: Translate ke English jika bukan English
        english_query = self.translate_to_english(user_query, detected_lang)
        
        # Step 3: Semantic search dengan query English
        print(f"Mencari context relevan (top-{self.top_k})...")
        contexts = self._search_context(english_query)
        
        # Step 4: Format prompt dengan instruksi yang lebih baik
        lang_instruction = f"IMPORTANT: Respond ONLY in the same language as this original user question: '{user_query}'. Do not provide multiple language versions or translations."
        
        # Buat prompt yang lebih eksplisit tentang kapan harus merekomendasikan course
        prompt = f"""{self.system_prompt}

{self._format_context(contexts)}

{self._format_history()}

{lang_instruction}

Current User Question: {user_query}

Please provide a helpful response based on the context above. At the end of your response, if you think the user would benefit from seeing the full course material on one of these topics, include a recommendation like this exactly:

RECOMMENDED_COURSE: [course_id]

Otherwise, don't include this line."""

    try:
        print("CS Helper bot is answering...")
        response = self.gemini_model.generate_content(prompt)
        bot_response = response.text.strip()
        
        # Cek apakah ada rekomendasi course dari Gemini
        recommended_course = None
        if "RECOMMENDED_COURSE:" in bot_response:
            # Ekstrak course_id dari response
            parts = bot_response.split("RECOMMENDED_COURSE:")
            bot_response = parts[0].strip()
            recommended_course = parts[1].strip()
            
            # Hapus kurung siku jika ada
            recommended_course = recommended_course.replace("[", "").replace("]", "").strip()
            
            # Verifikasi course_id valid
            if not any(course['id'] == recommended_course for course in COURSES_DATA):
                recommended_course = None
        
        # Update history dengan bahasa asli user
        self._update_history(user_query, bot_response)
        return bot_response, recommended_course, detected_lang
            
    except Exception as e:
        error_msg = f"Maaf, terjadi error: {str(e)}"
        # Translate error message ke bahasa user jika perlu
        if detected_lang != 'en':
            error_msg = self.translator.translate(error_msg, dest=detected_lang).text
        return error_msg, None, detected_lang

    def chat(self, user_query):
        """
        Fungsi interaktif chatbot dengan multilingual support:
        - Terima pertanyaan user dalam bahasa apapun
        - Proses dengan alur multilingual
        - Cetak hasil dan context yang ditemukan
        """
        print(f"\nUser: {user_query}")
        bot_response, contexts, detected_lang = self.generate_response(user_query)
        print(f"Bot: {bot_response}")

        # Debug: tampilkan context yang ditemukan
        print(f"\n[DEBUG] Detected Language: {detected_lang}")
        print("[DEBUG] Context yang ditemukan:")
        for i, ctx in enumerate(contexts, 1):
            print(f"  {i}. {ctx['tag']} (similarity: {ctx['similarity']:.4f})")

        return bot_response

    # Fungsi untuk clear chat history
    def clear_history(self):
        self.chat_history = []
        print("Chat history sudah dihapus.")

    # Fungsi untuk melihat seluruh chat history
    def show_history(self):
        if not self.chat_history:
            print("Belum ada chat history.")
            return
        print("\n=== CHAT HISTORY ===")
        for i, h in enumerate(self.chat_history, 1):
            print(f"{i}. User: {h['user']}")
            print(f"   Bot: {h['assistant']}\n")

    # Fungsi untuk melihat bahasa yang didukung
    def show_supported_languages(self):
        """Menampilkan daftar bahasa yang didukung oleh Google Translate"""
        print("\n=== SUPPORTED LANGUAGES ===")
        for code, name in LANGUAGES.items():
            print(f"{code}: {name}")
            
            
# if __name__ == "__main__":
#     print("=" * 60)
#     print("INITIALIZING RAG CHATBOT (VERSI INTERAKTIF)")
#     print("=" * 60)

#     # Hanya initialize chatbot jika data dari block sebelumnya sudah siap
#     chatbot = RAGChatbot(
#         faq_file="faq.json", 
#         model_path="best_embedding_model", 
#         top_k=3, 
#         max_history=10,
#         temperature=0.7,
#         top_p=0.9,
#         top_k_gen=40
#     )
#     print("\nChatbot siap digunakan!")

#     # Ganti test_queries dengan loop interaktif
#     print("\nKetik 'exit' untuk berhenti.\n")
#     while True:
#         user_input = input("You: ")
#         if user_input.strip().lower() == 'exit':
#             print("Chatbot: Goodbye! Terima kasih sudah menggunakan chatbot ini.")
#             break
#         chatbot.chat(user_input)
#         print("-" * 40)

#     chatbot.show_history()
