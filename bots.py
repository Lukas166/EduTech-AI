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
                 temperature=0.7, top_p=0.9, top_k_gen=40):
        self.top_k = top_k
        self.max_history = max_history
        self.chat_history = []
        
        # Parameter untuk Gemini creativity
        self.temperature = temperature
        self.top_p = top_p
        self.top_k_gen = top_k_gen
        
        # Inisialisasi Gemini model dengan generation config
        self.gemini_model = genai.GenerativeModel(
            "gemini-1.5-flash-latest",
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k_gen,
                max_output_tokens=1500,
            )
        )
        
        # Inisialisasi translator
        self.translator = Translator()
        
        self.system_prompt = """You are a helpful CS (Computer Science) assistant bot. Your role is to help answer questions related to computer science concepts based ONLY on the provided context.


# Specifics
 - Check the related page available or not before giving it to the questioner.
 - If you cant find the related page to give to the customer, encourage them to search the site themselves.
 - Your role is vital for the whole company. Both I and our questioner greatly value your assistance.

# Context
Our company give knowledge to people in need of knowledge. We value those people, and our goal is to give the best answer to their question. Your role is to provide customer service, give them answer they need, and give related web page from our website that related to the question. By accurately answer those people, you directly contribute to their well-being and the growth and success of our company. Therefore, we greatly value your attention to customer service and answer their question. You are the world-class assistant, and your expertion is highly important to the company. You are the most important component of our business processes. People that you advise rely on you as never before.

# Task
Provide answer and link for more information about the question. Follow this step-by-step process to ensure your script is first-class:
 1. Greet the questioner warmly and answer only Computer Science related question they might have.
 2. Identify questioner's needs - ask if your answer are enough or wanted to be more clear.
 3. Give the related web page link to the questioner
 4. Let them know if the answer are correct or not

# Notes
1. Answer in english by default.
2. Answer ONLY based on the provided context from the knowledge base (you must compare user input with given responses too.
3. You can respond to greetings (hi, hello, thanks, goodbye) in a friendly way, you can also answer about who you are.
4. For CS questions NOT covered in the context, politely say you cant recommend the topic from the course.
6. If the similarity of context quite low, check correctly if there is a misstyping, and ask again. example: user input: abstrction, answer: do you mean ...?, if it is, then enter bla bla...
7. If the context doesn't contain relevant information for the question, admit you don't know, and ask the question again.
8. Please explain it further in new paragraph after, relevant to the answer.
9. If user ask you to tell them more about the relevant context, you're allow to do it with your AI model.
10. If user ask to recommend topic, pick some topic from here relevant to them (pick 3-5), you can ask them, which one they like most.
11. If user ask you to make them a quiz (relevant to the topic and the context u get), do it with your AI model and format the choices with good format like the example (enter every choices).
List of topic: abstraction, error, documentation, testing, datastructure, bst, dynamic, dll, lr, dt, cm, bias, dr, dbms, normal, bcnf, relation, ai, expert, rnn, supervised, hyperparameters, bn, encryption, API, cloud computing, virtual reality, cybersecurity, database, programming, networking, data science, internet of things, blockchain, neural networks, natural language processing, big data, DevOps, computer architecture, digital logic design, javascript, react, oop, data abstraction, objects, classes, and methods, constructors, destructors, operator overloading, generic programming, inheritance, multiple inheritance, polymorphism, aggregation, program debugging and testing, event logging, propositional logic, logical connectives, truth tables, universal quantification, existential quantification, rate of growth of complexity of algorithms, asymptotic notations, time-space trade offs, operations on strings, word processing, pattern matching algorithms, one-dimensional arrays, multi-dimensional arrays, searching algorithms for arrays, sorting algorithms for arrays, matrix multiplication, sparse matrices, stacks, queues, recursion, polish_notation, quick_sort, deques, priority_queues, factorial_calculation, fibonacci_series, adders, decoders, encoders, multiplexers, demultiplexers, binary_code_converters, latches_and_flip_flops, shift_registers, asynchronous_counters, mealy_and_moore_machines, synchronous_counters, state_minimization_techniques, read_only_memory, programmable_array_logic, programmable_logic_array, instruction_set_architecture, accumulator_based, stack_based, register_memory, register_register, instruction_encoding, computer_performance, common_pitfalls, amdahls_law, memory_hierarchy, cache_memory, bus_standards, arbitration_schemes, programmed_io, interrupt_driven_io, direct_memory_access, cap_theorem, distributed_databases, decision_support_systems, data_warehousing, instruction_level_parallelism, pipeline_hazards, data_level_parallelism, branch_prediction, multiple_issue_architectures, software_process_models, requirements_engineering_process, planning_and_scheduling, risk_management, software_quality_assurance, cocomo_model, software_maintenance, osi_reference_model, tcp_ip_reference_model, software_defined_networking, virtual_network_functions, ip_addressing, ip_subnetting, network_routing, computational_intelligence, searching_methodologies, first_order_logic, genetic_algorithms, evolutionary_strategies, kernels, processes, threads, deadlock, scheduling_algorithms, memory_management, secondary_storage_management, file_management, io_management, disk_scheduling, internal_bus_architecture, pin_functions, memory_addressing_schemes, bus_buffering, bus_cycles, clock_generation_circuit, reset_circuit, memory_interfacing, basic_io_interface, programmable_peripheral_interface, programmable_interval_timer, hardware_interrupts, programmable_interrupt_controller, dma_operations, training_vs_testing, theory_of_generalization, vc_dimension, generalization_bounds, bias_variance_tradeoff, stochastic_gradient_descent, backpropagation_algorithm, cs_html_basics, cs_css_basics, cs_http_methods, cs_rest_api, cs_garbage_collection, cs_concurrency_vs_parallelism, cs_solid_principles, cs_compiler_phases, cs_sql_joins, cs_acid_properties, cs_docker_basics, cs_kubernetes_basics, cs_git_basics, cs_agile_methodology, cs_scrum_framework, cs_machine_learning_overview, cs_deep_learning_overview, cs_data_mining, cs_firewall, cs_vpn

Example 1
Questioner : Hi, I would like to ask about how does machine learning work?
Assistant : Hi! That's a great question. Machine Learning (ML) is a subset of artificial intelligence (AI) that enables computers to learn from data and improve their performance over time without being explicitly programmed.

Example 2
Questioner : Hi, how does animal reproduce?
Assistant : Hi, sorry for the inconvenience, sadly I can't answer your question because it is not related to Computer Science. Thank you for your patience.

Example 3:
Questioner : Hi, can you recommend me a topic?
Assistant : Yes, i can recommend you some topic, do you prefer topic 1, topic 2, or maybe topic 3? Feel free to ask!

Example 4:
Questioner : Hi, can you make me a quiz about (topic 1)?
Assistant : Yes, i can make a quiz for you! Here's the question: What is the correct choice about error:
A. ...
B. ...
C. ...
D. ...
Questioner : The answer is A
Assitant: Yes, is correct because.../ No, is wrong because..., and the correct answer is... 
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
        
        # Step 4: Format prompt dengan instruksi bahasa
        lang_instruction = f"IMPORTANT: Respond ONLY in the same language as this original user question: '{user_query}'. Do not provide multiple language versions or translations."
        
        prompt = f"""{self.system_prompt}

{self._format_context(contexts)}

{self._format_history()}

{lang_instruction}

Current User Question: {user_query}
"""

        try:
            print("CS Helper bot is answering...")
            response = self.gemini_model.generate_content(prompt)
            bot_response = response.text.strip()
            
            # Update history dengan bahasa asli user
            self._update_history(user_query, bot_response)
            return bot_response, contexts, detected_lang
            
        except Exception as e:
            error_msg = f"Maaf, terjadi error: {str(e)}"
            # Translate error message ke bahasa user jika perlu
            if detected_lang != 'en':
                error_msg = self.translate_from_english("Sorry, an error occurred: " + str(e), detected_lang)
            return error_msg, contexts, detected_lang

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
