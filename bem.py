import streamlit as st
import os
import time
import re
import google.generativeai as genai
from langchain.llms.base import LLM
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# --- Konfigurasi halaman ---
st.set_page_config(page_title="Chatbot BEM FMIPA", layout="wide")
st.markdown("<h1 style='text-align: center;'>🎓 Chatbot BEM FMIPA Universitas Udayana</h1>", unsafe_allow_html=True)

# --- Konfigurasi Gemini ---
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --- Class Gemini kompatibel LangChain ---
class ChatGemini(LLM):
    model: str = "gemini-2.5-flash"
    temperature: float = 0.2

    def _call(self, prompt: str, stop=None) -> str:
        try:
            model = genai.GenerativeModel(self.model)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"[Error Gemini] {str(e)}"

    @property
    def _llm_type(self) -> str:
        return "gemini_custom"

# --- Format jawaban agar rapi ---
def format_answer(text):
    formatted = re.sub(r'(\d+)\.\s*', r'\n\1. ', text)
    formatted = re.sub(r'(\d+\.\s[^\n]+)', r'\1\n', formatted)
    return formatted.strip()

# --- Load PDF vectorstore ---
@st.cache_resource(show_spinner=False)
def load_pdf_vectorstore(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250)
    chunks = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
    return retriever

# --- State Management ---
if "messages_bem" not in st.session_state:
    st.session_state.messages_bem = []
if "messages_general" not in st.session_state:
    st.session_state.messages_general = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "mode" not in st.session_state:
    st.session_state.mode = "bem"

# --- Muat dataset otomatis ---
default_pdf = "dataset_bem.pdf"
if os.path.exists(default_pdf) and st.session_state.retriever is None:
    with st.spinner("📚 Membaca dataset BEM FMIPA 2025..."):
        st.session_state.retriever = load_pdf_vectorstore(default_pdf)
        st.success("✅ Dataset BEM FMIPA berhasil dimuat!")

# --- Styling Chat ---
st.markdown("""
<style>
.chat-bubble-user {
    background-color:#DCF8C6;
    border-radius:12px;
    padding:8px 12px;
    margin:5px;
    max-width:80%;
    float:right;
    clear:both;
}
.chat-bubble-bot {
    background-color:#EAEAEA;
    border-radius:12px;
    padding:8px 12px;
    margin:5px;
    max-width:80%;
    float:left;
    clear:both;
}
</style>
""", unsafe_allow_html=True)

# --- Fungsi utama handle pertanyaan ---
def handle_question(question):
    llm = ChatGemini(model="gemini-2.5-flash", temperature=0.2)
    mode = st.session_state.mode

    # --- Mode umum ---
    if mode == "general":
        prompt = f"""
Kamu adalah asisten AI untuk menjawab pertanyaan umum dengan bahasa alami.
Jawablah secara akurat, ramah, dan sopan.

Pertanyaan:
{question}
"""
        answer = llm._call(prompt)
        answer = format_answer(answer)
        st.session_state.messages_general.append({"role": "user", "content": question})
        st.session_state.messages_general.append({"role": "assistant", "content": answer})
        return

    # --- Mode BEM ---
    retriever = st.session_state.retriever
    context = ""
    if retriever:
        docs = retriever.get_relevant_documents(question)
        context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
Kamu adalah Chatbot BEM FMIPA Universitas Udayana.
Gunakan seluruh konteks di bawah ini untuk menjawab dengan lengkap dan akurat.
Jika tidak ditemukan, katakan dengan sopan bahwa informasinya belum tersedia.

Konteks:
{context}

Pertanyaan:
{question}
"""
    answer = llm._call(prompt)
    answer = format_answer(answer)
    st.session_state.messages_bem.append({"role": "user", "content": question})
    st.session_state.messages_bem.append({"role": "assistant", "content": answer})

# --- Tombol Mode ---
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("📘 Aktifkan Mode BEM"):
        st.session_state.mode = "bem"
        st.session_state.messages_bem.append({
            "role": "assistant",
            "content": "📘 Sekarang kamu berada di mode BEM — tanya apa pun seputar BEM FMIPA!"
        })
        st.rerun()

# --- Quick Questions (dijamin ada di PDF) ---
if st.session_state.mode == "bem":
    quick_questions = [
        "Apa tugas harian Bidang Advokasi dan Kesejahteraan Mahasiswa?",
        "Apa program kerja Bidang Minat dan Bakat?",
        "Apa tugas harian Bidang Komunikasi dan Informasi?",
        "Apa peran Bidang Pendidikan dan Penalaran?",
        "Apa tugas Badan Pengurus Harian?",
        "Apa tugas Badan Pengelola Administrasi?",
        "Apa program kerja Bidang Sosial dan Pengabdian?",
        "Apa tugas Bidang Ekonomi Kreatif?",
        "Apa tugas Bidang Kaderisasi dan Pengembangan Sumber Daya Mahasiswa?"
    ]

    st.markdown("### 💡 Contoh pertanyaan siap klik:")
    cols = st.columns(3)
    for i, q in enumerate(quick_questions):
        if cols[i % 3].button(q):
            handle_question(q)
            st.rerun()

with col2:
    if st.button("🌐 Aktifkan Mode Umum"):
        st.session_state.mode = "general"
        st.session_state.messages_general.append({
            "role": "assistant",
            "content": "🌐 Sekarang kamu berada di mode umum — tanya hal-hal di luar BEM FMIPA!"
        })
        st.rerun()

with col3:
    if st.button("🧹 Hapus Semua Riwayat"):
        st.session_state.messages_bem = []
        st.session_state.messages_general = []
        st.success("Semua riwayat chat berhasil dihapus!")
        st.rerun()

# --- Tampilan Riwayat Chat ---
if st.session_state.messages_bem:
    st.markdown("## 📘 Percakapan Mode BEM")
    for msg in st.session_state.messages_bem:
        role_class = "chat-bubble-user" if msg["role"] == "user" else "chat-bubble-bot"
        st.markdown(f"<div class='{role_class}'>{msg['content']}</div>", unsafe_allow_html=True)

if st.session_state.messages_general:
    st.markdown("## 🌐 Percakapan Mode Umum")
    for msg in st.session_state.messages_general:
        role_class = "chat-bubble-user" if msg["role"] == "user" else "chat-bubble-bot"
        st.markdown(f"<div class='{role_class}'>{msg['content']}</div>", unsafe_allow_html=True)

# --- Input Chat ---
user_input = st.chat_input("Ketik pertanyaan kamu di sini...")
if user_input:
    handle_question(user_input)
    st.rerun()

st.markdown("---")
st.caption("🤖 Chatbot BEM FMIPA | Dual Mode | Gemini RAG")