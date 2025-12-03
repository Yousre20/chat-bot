# --- 0. FORCE INSTALL (The "Nuclear" Fix) ---
import os
import subprocess
import sys

# Function to install libraries if missing
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Check if langchain is installed, if not, force install EVERYTHING
try:
    import langchain
    import pysqlite3
except ImportError:
    print("⚠️ Libraries missing. Force installing... (This takes 1 minute)")
    install("streamlit")
    install("pandas")
    install("langchain")
    install("langchain-community")
    install("langchain-huggingface")
    install("langchain-chroma")
    install("sentence-transformers")
    install("chromadb")
    install("python-dotenv")
    install("pysqlite3-binary")
    install("huggingface-hub")
    print("✅ Installation complete.")

# --- 1. Database Fix ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# --- 2. Imports ---
import streamlit as st
import pandas as pd
import shutil

from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA # Using the simplest, most stable chain
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

# --- 3. Page Config ---
st.set_page_config(page_title="Banque Masr AI", page_icon="🏦")
st.title("🏦 Banque Masr Assistant")

# --- 4. Constants ---
# Using Flan-T5 because it works 100% on the free API without "Task" errors
REPO_ID = "google/flan-t5-large"
CHROMA_PATH = "./chroma_db_data"

# --- 5. Secrets ---
if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
else:
    api_key = st.sidebar.text_input("Enter Hugging Face Token", type="password")
    if api_key:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
    else:
        st.warning("Please add your Hugging Face Token.")
        st.stop()

# --- 6. Load Data ---
@st.cache_resource
def load_db():
    # 1. Locate File
    files = ["data/BankFAQs.csv", "BankFAQs.csv"]
    file_path = next((f for f in files if os.path.exists(f)), None)
    
    if not file_path:
        st.error("❌ 'BankFAQs.csv' not found.")
        return None

    # 2. Reset DB (Fixes 'no such table' error)
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # 3. Process Data
    bank = pd.read_csv(file_path)
    bank["content"] = bank.apply(lambda row: f"Q: {row['Question']}\nA: {row['Answer']}", axis=1)
    docs = [Document(page_content=row["content"]) for _, row in bank.iterrows()]

    # 4. Vector DB
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_PATH)

@st.cache_resource
def load_llm():
    return HuggingFaceEndpoint(
        repo_id=REPO_ID, 
        task="text2text-generation", 
        temperature=0.1,
        max_new_tokens=512
    )

# --- 7. App Logic ---
with st.spinner("Starting AI (This might take 2 mins first time)..."):
    db = load_db()
    llm = load_llm()

if not db or not llm:
    st.stop()

# --- 8. The Chain (RetrievalQA - The most stable one) ---
template = """Answer the question based ONLY on the context below.

Context: {context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 2}),
    chain_type_kwargs={"prompt": PROMPT}
)

# --- 9. Chat UI ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = qa_chain.invoke(prompt)
                output = response["result"] # RetrievalQA uses 'result'
                st.write(output)
                st.session_state.messages.append({"role": "assistant", "content": output})
            except Exception as e:
                st.error(f"Error: {e}")
