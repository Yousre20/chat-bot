import streamlit as st
import pandas as pd
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Banque Masr AI", page_icon="🏦")
st.title("🏦 Banque Masr Assistant (Stable Version)")

# Constants
DATA_FILE = "data/BankFAQs.csv" # Ensure this path matches your GitHub
MODEL_ID = "google/flan-t5-large" # The most reliable free model for RAG

# --- 2. AUTHENTICATION ---
# Check secrets first, then sidebar
if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
    HF_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
else:
    HF_TOKEN = st.sidebar.text_input("Enter Hugging Face Token", type="password")

if not HF_TOKEN:
    st.warning("⚠️ Please enter your Hugging Face Token to continue.")
    st.stop()

# --- 3. THE BRAIN (Load & Index Data) ---
@st.cache_resource
def initialize_engine():
    """
    Loads CSV, converts to numbers (embeddings), and builds a FAISS index.
    This runs once and stays in memory. Zero database files to corrupt.
    """
    # A. Check for file
    # We check root folder AND data folder to be safe
    final_path = None
    if os.path.exists(DATA_FILE):
        final_path = DATA_FILE
    elif os.path.exists("BankFAQs.csv"):
        final_path = "BankFAQs.csv"
    
    if not final_path:
        st.error(f"❌ Critical Error: Could not find 'BankFAQs.csv'. Please upload it to GitHub.")
        return None, None, None

    # B. Load Data
    try:
        df = pd.read_csv(final_path)
        # Create a clean list of text chunks
        df['combined'] = "Question: " + df['Question'] + "\nAnswer: " + df['Answer']
        documents = df['combined'].tolist()
    except Exception as e:
        st.error(f"❌ Error reading CSV: {e}")
        return None, None, None

    # C. Load Embedding Model (The Translator)
    # Using a small, fast model ideal for CPU
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    
    # D. Create Embeddings
    embeddings = encoder.encode(documents)
    
    # E. Build FAISS Index (The Search Engine)
    # Dimension 384 is standard for MiniLM
    index = faiss.IndexFlatL2(384) 
    index.add(embeddings)
    
    return index, documents, encoder

# --- 4. THE FUNCTIONS ---

def retrieve_info(query, index, documents, encoder, k=3):
    """Finds the top 3 most relevant FAQs"""
    # Convert query to vector
    query_vector = encoder.encode([query])
    
    # Search FAISS
    # D = distances, I = indices (row numbers)
    D, I = index.search(query_vector, k)
    
    # Fetch actual text
    results = [documents[i] for i in I[0]]
    return "\n\n".join(results)

def generate_answer(context, question):
    """Sends prompt to Hugging Face"""
    client = InferenceClient(token=HF_TOKEN)
    
    system_prompt = """You are a helpful banking assistant for Banque Masr. 
Answer the question based strictly on the context provided. 
If the answer is not in the context, say 'I cannot find that information in my records'."""
    
    prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"

    try:
        response = client.text_generation(
            model=MODEL_ID,
            prompt=prompt,
            max_new_tokens=512,
            temperature=0.1, # Keep it factual
            seed=42
        )
        return response
    except Exception as e:
        return f"⚠️ Model Error: {str(e)}"

# --- 5. MAIN APP UI ---

# Load the engine (Cached)
with st.spinner("Building Knowledge Base..."):
    index, documents, encoder = initialize_engine()

if index is None:
    st.stop()

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome to Banque Masr! How can I help you today?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User Input
if prompt := st.chat_input("Ask about loans, accounts, or cards..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Process
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # 1. Retrieve
            context = retrieve_info(prompt, index, documents, encoder)
            
            # 2. Generate
            answer = generate_answer(context, prompt)
            
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
