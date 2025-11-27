# chatbot.py — HF Inference API version (no torch, Streamlit Cloud safe)
import streamlit as st
import pandas as pd
import numpy as np
from huggingface_hub import InferenceClient
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Banque Misr Chatbot", page_icon="🤖")

# -------------------------
# Config
# -------------------------
CSV_PATH = "BankFAQs.csv"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL = "google/flan-t5-small"   # صغير مناسب للـ Cloud
TOP_K_DEFAULT = 1

# -------------------------
# HF Client
# -------------------------
HF_TOKEN = st.secrets.get("HF_TOKEN", None)
if not HF_TOKEN:
    st.error("HF_TOKEN missing in Streamlit secrets. Go to Settings → Secrets and add HF_TOKEN.")
    st.stop()

client = InferenceClient(token=HF_TOKEN)

# -------------------------
# Load FAQ data
# -------------------------
@st.cache_data(show_spinner=False)
def load_faqs(path):
    df = pd.read_csv(path)
    df["content"] = df.apply(lambda r: f"Question: {r['Question']}\nAnswer: {r['Answer']}", axis=1)
    return df

# -------------------------
# Build embeddings safely
# -------------------------
@st.cache_data(show_spinner=False)
def build_corpus_embeddings(df, model_name):
    embeddings = []
    for t in df["Question"].tolist():
        out = client.feature_extraction(model=model_name, inputs=t)
        arr = np.array(out)
        if arr.ndim > 1:
            arr = arr.flatten()
        embeddings.append(arr)
    return np.vstack(embeddings)

# -------------------------
# Retrieve best context
# -------------------------
def retrieve_best_context(query, corpus_emb, df, top_k=TOP_K_DEFAULT):
    q_emb = np.array(client.feature_extraction(model=EMBED_MODEL, inputs=query))
    if q_emb.ndim > 1:
        q_emb = q_emb.flatten().reshape(1, -1)
    sims = cosine_similarity(q_emb, corpus_emb)[0]
    idxs = np.argsort(sims)[::-1][:top_k]
    return "\n\n".join(df.iloc[i]["content"] for i in idxs)

# -------------------------
# Streamlit UI
# -------------------------
st.title("🤖 Banque Misr Chatbot (HF Inference)")
st.write("Ask about banking products, loans, accounts, or general finance questions.")

df = load_faqs(CSV_PATH)
corpus_emb = build_corpus_embeddings(df, EMBED_MODEL)

# Sidebar options
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of retrieved contexts (top_k)", 1, 5, TOP_K_DEFAULT)
    show_context = st.checkbox("Show retrieved context", False)
    st.markdown("---")
    st.markdown("💡 Notes:")
    st.markdown("- HF Inference API handles embeddings & generation; no need for torch locally.")
    st.markdown("- Use small models for faster response (FLAN-T5-small).")

# Chat UI
user_input = st.text_input("You:", "")
if st.button("Send") or user_input:
    if not user_input.strip():
        st.warning("Write a question first.")
    else:
        with st.spinner("Retrieving context..."):
            context = retrieve_best_context(user_input, corpus_emb, df, top_k=top_k)
        
        if show_context:
            st.markdown("**Retrieved context:**")
            st.code(context)
        
        prompt = f"""
You are an expert Finance Chatbot working for a bank. Answer concisely using the context below.

Context: {context}
Customer Question: {user_input}
Answer:
"""
        with st.spinner("Generating answer..."):
            out = client.text_generation(model=GEN_MODEL, inputs=prompt, parameters={"max_new_tokens":200, "temperature":0.3})
            # Safely parse output
            if isinstance(out, str):
                generated = out
            elif isinstance(out, list):
                # list of dicts
                generated = out[0].get("generated_text", str(out))
            elif isinstance(out, dict):
                generated = out.get("generated_text", str(out))
            else:
                generated = str(out)
        st.markdown(f"**Chatbot:** {generated}")
