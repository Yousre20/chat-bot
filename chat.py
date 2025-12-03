# --- 1. Database Fix (Must be at the very top) ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# --- 2. Imports ---
import streamlit as st
import pandas as pd
import os
import shutil

# Modern LangChain Imports
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# --- 3. Page Config ---
st.set_page_config(page_title="Banque Masr AI", page_icon="🏦")
st.title("🏦 Banque Masr Assistant")

# --- 4. Constants ---
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
def load_resources():
    # 1. Locate File
    files = ["data/BankFAQs.csv", "BankFAQs.csv"]
    file_path = next((f for f in files if os.path.exists(f)), None)
    
    if not file_path:
        st.error("❌ 'BankFAQs.csv' not found.")
        return None, None

    # 2. Reset DB (Prevents corruption)
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # 3. Process Data
    bank = pd.read_csv(file_path)
    bank["content"] = bank.apply(lambda row: f"Q: {row['Question']}\nA: {row['Answer']}", axis=1)
    docs = [Document(page_content=row["content"]) for _, row in bank.iterrows()]

    # 4. Vector DB
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_PATH)
    
    # 5. Load LLM
    llm = HuggingFaceEndpoint(
        repo_id=REPO_ID, 
        task="text2text-generation", 
        temperature=0.1,
        max_new_tokens=512
    )
    
    return vector_db, llm

# --- 7. App Logic ---
with st.spinner("Starting AI..."):
    vector_db, llm = load_resources()

if not vector_db or not llm:
    st.stop()

# --- 8. The New Chain (Modern Method) ---
template = """Answer the question based ONLY on the context below.

Context: {context}

Question: {input}

Answer:"""

PROMPT = PromptTemplate.from_template(template)

# Create the chain using the new 'create' functions
question_answer_chain = create_stuff_documents_chain(llm, PROMPT)
rag_chain = create_retrieval_chain(vector_db.as_retriever(search_kwargs={"k": 2}), question_answer_chain)

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
                # The new chain expects 'input', not 'question'
                response = rag_chain.invoke({"input": prompt})
                output = response["answer"]
                st.write(output)
                st.session_state.messages.append({"role": "assistant", "content": output})
            except Exception as e:
                st.error(f"Error: {e}")
