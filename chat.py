# --- 1. Database Hack ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# --- 2. Imports ---
import streamlit as st
import pandas as pd
import os
import shutil
import chromadb
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

# --- 3. Configuration ---
st.set_page_config(page_title="Banque Masr AI", page_icon="🏦")
st.title("🏦 Banque Masr Assistant (Pure Python)")

# Constants
REPO_ID = "google/flan-t5-large"
CHROMA_PATH = "./chroma_db_data"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- 4. Secrets Handling ---
if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
    HF_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
else:
    HF_TOKEN = st.sidebar.text_input("Enter Hugging Face Token", type="password")
    if not HF_TOKEN:
        st.warning("Please add your Hugging Face Token.")
        st.stop()

# --- 5. Core Functions ---

@st.cache_resource
def setup_vector_db():
    files = ["data/BankFAQs.csv", "BankFAQs.csv"]
    file_path = next((f for f in files if os.path.exists(f)), None)
    
    if not file_path:
        st.error("❌ 'BankFAQs.csv' not found.")
        return None, None

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    # --- THE FIX IS HERE ---
    # We catch generic Exception to handle both ValueError and NotFoundError
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
    client = InferenceClient(token=HF_TOKEN)
    
    prompt = f"""Answer based on context.

Context:
{context}

Question: 
{question}

Answer:"""

    try:
        # Use the official text_generation method
        response = client.text_generation(
            model=REPO_ID,
            prompt=prompt,
            max_new_tokens=512,
            temperature=0.1,
            seed=42
        )
        return response
    except Exception as e:
        return f"Error: {e}"

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
