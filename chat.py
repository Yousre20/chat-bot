# chatbot.py (HF Inference API version — no torch required)
import streamlit as st
import pandas as pd
import numpy as np
from huggingface_hub import InferenceClient
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Banque Misr Chatbot", page_icon="🤖")

CSV_PATH = "BankFAQs.csv"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL = "google/flan-t5-small"   # أو أي نموذج نصي متاح في HF Inference

# HF client from Streamlit secrets
HF_TOKEN = st.secrets.get("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", None)
if not HF_TOKEN:
    st.error("HF_TOKEN missing in Streamlit secrets. Go to Settings → Secrets and add HF_TOKEN.")
    st.stop()

client = InferenceClient(token=HF_TOKEN)

@st.cache_data(show_spinner=False)
def load_faqs(path):
    df = pd.read_csv(path)
    df["content"] = df.apply(lambda r: f"Question: {r['Question']}\nAnswer: {r['Answer']}", axis=1)
    return df

@st.cache_resource(show_spinner=False)
def build_corpus_embeddings(df, model_name):
    texts = df["Question"].tolist()
    # call HF inference embedding endpoint in batch
    embeddings = []
    for t in texts:
        out = client.feature_extraction(model=model_name, inputs=t)
        # out may be list of floats
        embeddings.append(np.array(out))
    return np.vstack(embeddings)

def retrieve_best_context(query, corpus_emb, df, top_k=1):
    q_emb = np.array(client.feature_extraction(model=EMBED_MODEL, inputs=query))
    sims = cosine_similarity(q_emb.reshape(1, -1), corpus_emb)[0]
    idxs = np.argsort(sims)[::-1][:top_k]
    return "\n\n".join(df.iloc[i]["content"] for i in idxs)

# UI
st.title("🤖 Banque Misr Chatbot (HF Inference)")
st.write("Ask about banking products, loans, accounts, or general finance questions.")

df = load_faqs(CSV_PATH)
corpus_emb = build_corpus_embeddings(df, EMBED_MODEL)

user_input = st.text_input("You:", "")
top_k = st.sidebar.slider("top_k", 1, 5, 1)
show_context = st.sidebar.checkbox("Show retrieved context", False)

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
            # Out format varies; try to extract text
            if isinstance(out, str):
                generated = out
            elif isinstance(out, list):
                # sometimes returns list of dicts
                generated = out[0].get("generated_text", str(out))
            elif isinstance(out, dict):
                generated = out.get("generated_text", str(out))
            else:
                generated = str(out)
        st.markdown(f"**Chatbot:** {generated}")
