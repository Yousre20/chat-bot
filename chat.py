# --- 1. Database Hack (Must be at the very top) ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# --- 2. Imports ---
import streamlit as st
import pandas as pd
import os
import shutil

# Standard Libraries
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.documents import Document

# --- 3. Page Config ---
st.set_page_config(page_title="Banque Masr AI", page_icon="🏦")
st.title("🏦 Banque Masr Assistant")

# --- 4. Constants ---
REPO_ID = "google/flan-t5-large" # Stable Model
CHROMA_PATH = "./chroma_db_data"

# --- 5. Secrets ---
if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
else:
    st.warning("Please add your Hugging Face Token to Streamlit Secrets.")
    st.stop()

# --- 6. Load Data ---
@st.cache_resource
def load_data():
    # Look for the file in probable locations
    files = ["data/BankFAQs.csv", "BankFAQs.csv"]
    file_path = next((f for f in files if os.path.exists(f)), None)

    if not file_path:
        st.error("❌ 'BankFAQs.csv' not found on GitHub!")
        return None

    # Clean DB to prevent "no table" errors
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Process Data
    bank = pd.read_csv(file_path)
    bank["content"] = bank.apply(lambda row: f"Q: {row['Question']}\nA: {row['Answer']}", axis=1)
    docs = [Document(page_content=row["content"]) for _, row in bank.iterrows()]

    # Build Vector DB
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_PATH)
    return db

# --- 7. Load Model ---
@st.cache_resource
def load_llm():
    return HuggingFaceEndpoint(
        repo_id=REPO_ID, 
        task="text2text-generation", 
        temperature=0.1
    )

# --- 8. App Logic ---
with st.spinner("Starting AI..."):
    db = load_data()
    llm = load_llm()

if db and llm:
    # Setup Memory
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True, 
            output_key='answer'
        )

    # Setup Chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        memory=st.session_state.memory
    )

    # Chat UI
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        st.chat_message("user").write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            response = chain.invoke({"question": prompt})
            st.write(response["answer"])
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

