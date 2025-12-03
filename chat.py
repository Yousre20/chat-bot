# --- 1. Database Hack (Required for Streamlit Cloud) ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# --- 2. Imports ---
import streamlit as st
import pandas as pd
import os
import shutil
import requests
import chromadb
from sentence_transformers import SentenceTransformer

# --- 3. Configuration ---
st.set_page_config(page_title="Banque Masr AI", page_icon="🏦")
st.title("🏦 Banque Masr Assistant")

# Constants
REPO_ID = "google/flan-t5-large"
# FIXED: Updated URL from 'api-inference' to 'router'
API_URL = f"https://router.huggingface.co/models/{REPO_ID}"
CHROMA_PATH = "./chroma_db_data"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- 4. Secrets Handling (Side Bar) ---
with st.sidebar:
    st.header("Settings")
    user_token = st.text_input("Hugging Face Token", type="password")

    if user_token:
        HF_TOKEN = user_token
    elif "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
        HF_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    else:
        st.warning("⚠️ Please enter your token here to start.")
        st.stop()

# --- 5. Core Functions ---

@st.cache_resource
def setup_vector_db():
    files = ["data/BankFAQs.csv", "BankFAQs.csv"]
    file_path = next((f for f in files if os.path.exists(f)), None)
    
    if not file_path:
        st.error("❌ 'BankFAQs.csv' not found. Please upload it to GitHub.")
        return None, None

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        chroma_client.delete_collection("banque_masr")
    except Exception:
        pass 

    collection = chroma_client.create_collection(name="banque_masr")

    df = pd.read_csv(file_path)
    documents = df.apply(lambda row: f"Question: {row['Question']}\nAnswer: {row['Answer']}", axis=1).tolist()
    ids = [str(i) for i in range(len(documents))]
    metadatas = [{"class": str(row["Class"])} for _, row in df.iterrows()]

    embeddings = embedding_model.encode(documents).tolist()
    collection.add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)
    
    return collection, embedding_model

def get_context(query, collection, embedding_model):
    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=3)
    
    if results['documents']:
        return "\n\n".join(results['documents'][0])
    return ""

def query_llm(context, question):
    prompt = f"""Answer the question based strictly on the context below.

Context:
{context}

Question: 
{question}

Answer:"""

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.1,
            "return_full_text": False
        }
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        
        if response.status_code != 200:
            return f"⚠️ API Error {response.status_code}: {response.text}"
        
        output = response.json()
        if isinstance(output, list) and "generated_text" in output[0]:
            return output[0]["generated_text"]
        else:
            return str(output)
            
    except Exception as e:
        return f"Error connecting to AI: {e}"

# --- 6. App Logic ---

with st.spinner("Initializing AI Brain..."):
    collection, embedding_model = setup_vector_db()

if not collection:
    st.stop()

# --- 7. Chat UI ---

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome to Banque Masr! How can I help you today?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            context = get_context(prompt, collection, embedding_model)
            answer = query_llm(context, prompt)
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
