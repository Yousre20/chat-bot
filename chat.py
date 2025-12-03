import streamlit as st
import pandas as pd
import numpy as np
import faiss
import os
import requests
import json
import time
from sentence_transformers import SentenceTransformer

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Banque Masr AI", page_icon="🏦")
st.title("🏦 Banque Masr Assistant")

# Constants
DATA_FILE = "data/BankFAQs.csv"
# We use the standard API URL. If this fails, the error handler below will tell us why.
API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- 2. AUTHENTICATION ---
if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
    HF_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
else:
    HF_TOKEN = st.sidebar.text_input("Enter Hugging Face Token", type="password")

if not HF_TOKEN:
    st.warning("⚠️ Please enter your Hugging Face Token to continue.")
    st.stop()

# --- 3. THE BRAIN (FAISS Index) ---
@st.cache_resource
def initialize_engine():
    # Check paths
    final_path = None
    if os.path.exists(DATA_FILE):
        final_path = DATA_FILE
    elif os.path.exists("BankFAQs.csv"):
        final_path = "BankFAQs.csv"
    
    if not final_path:
        st.error("❌ Critical Error: Could not find 'BankFAQs.csv'.")
        return None, None, None

    # Load Data
    try:
        df = pd.read_csv(final_path)
        df['combined'] = "Question: " + df['Question'] + "\nAnswer: " + df['Answer']
        documents = df['combined'].tolist()
    except Exception as e:
        st.error(f"❌ Error reading CSV: {e}")
        return None, None, None

    # Embed
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = encoder.encode(documents)
    
    # Index
    index = faiss.IndexFlatL2(384) 
    index.add(embeddings)
    
    return index, documents, encoder

# --- 4. API HANDLER (The Fix) ---
def query_huggingface_api(payload):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        return response
    except Exception as e:
        return f"Connection Error: {e}"

def generate_answer(context, question):
    prompt = f"""Answer the question based strictly on the context below.

Context:
{context}

Question: 
{question}

Answer:"""

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.1
        }
    }

    # Attempt 1
    response = query_huggingface_api(payload)
    
    # Handle "Model Loading" (503) - Common on free tier
    if hasattr(response, 'status_code') and response.status_code == 503:
        with st.spinner("Model is loading (Cold Boot)... waiting 20 seconds..."):
            time.sleep(20)
            response = query_huggingface_api(payload)

    # Check for Errors
    if hasattr(response, 'status_code'):
        if response.status_code == 200:
            # Success!
            output = response.json()
            if isinstance(output, list) and 'generated_text' in output[0]:
                return output[0]['generated_text']
            elif isinstance(output, dict) and 'generated_text' in output:
                return output['generated_text']
            else:
                return str(output)
        elif response.status_code == 401:
            return "⚠️ Error 401: Unauthorized. Your Token is invalid. Please check Settings -> Secrets."
        elif response.status_code == 404:
            return "⚠️ Error 404: Model not found. Hugging Face might be down."
        else:
            return f"⚠️ API Error {response.status_code}: {response.text}"
            
    return str(response)

# --- 5. UI LOGIC ---

with st.spinner("Building Knowledge Base..."):
    index, documents, encoder = initialize_engine()

if index is None:
    st.stop()

# Chat
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome to Banque Masr!"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # 1. Search
            query_vector = encoder.encode([prompt])
            D, I = index.search(query_vector, 3)
            results = [documents[i] for i in I[0]]
            context = "\n\n".join(results)
            
            # 2. Ask API
            answer = generate_answer(context, prompt)
            
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
