import streamlit as st
import pandas as pd
import os
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# -------------------------
# Config
# -------------------------
CSV_PATH = "BankFAQs.csv"  # ضع BankFAQs.csv في جذر الريبو
USE_HF_INFERENCE_API = False  # لو True سيستخدم HuggingFace Inference API بدل تحميل الموديل محلياً
HF_MODEL = "google/flan-t5-small"  # أنصح بـ small للـ Streamlit Cloud، غيّري لو تحبي
LOCAL_MODEL = "google/flan-t5-small"  # استخدمي small لاقتصاد الموارد؛ غيّري إلى flan-t5-base لو عندك موارد
EMBEDDER_MODEL = "all-MiniLM-L6-v2"

st.set_page_config(page_title="Banque Misr Chatbot", page_icon="🤖", layout="centered")

# -------------------------
# Helpers: cache heavy resources
# -------------------------
@st.cache_data(show_spinner=False)
def load_faqs(path):
    if not os.path.exists(path):
        st.error(f"Missing file: {path}. Put BankFAQs.csv in the repo root.")
        return pd.DataFrame(columns=["Question", "Answer", "content"])
    bank = pd.read_csv(path)
    bank["content"] = bank.apply(lambda row: f"Question: {row['Question']}\nAnswer: {row['Answer']}", axis=1)
    return bank

@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer(EMBEDDER_MODEL)

@st.cache_resource(show_spinner=False)
def get_corpus_embeddings(bank_df, embedder):
    # Attempt to load precomputed embeddings file to save startup time
    emb_file = "corpus_embeddings.pt"
    if os.path.exists(emb_file):
        try:
            return torch.load(emb_file)
        except Exception:
            pass
    corpus = embedder.encode(bank_df["Question"].tolist(), convert_to_tensor=True, show_progress_bar=False)
    try:
        torch.save(corpus, emb_file)
    except Exception:
        # ignore saving failures on restricted environments
        pass
    return corpus

@st.cache_resource(show_spinner=False)
def get_qa_pipeline_local():
    # Load a smaller model to be safe on Streamlit Cloud
    return pipeline(
        "text2text-generation",
        model=LOCAL_MODEL,
        tokenizer=LOCAL_MODEL,
        device=0 if torch.cuda.is_available() else -1,
    )

# -------------------------
# Main app
# -------------------------
st.title("🤖 Banque Misr Chatbot — Finance FAQ")
st.write("Ask about banking products, loans, accounts, or general finance questions.")

bank = load_faqs(CSV_PATH)
if bank.empty:
    st.info("Upload `BankFAQs.csv` to the repository root (columns: Question, Answer).")
    st.stop()

embedder = get_embedder()
corpus_embeddings = get_corpus_embeddings(bank, embedder)

# If using HF Inference API (remote), we will call inference API instead of local pipeline
if USE_HF_INFERENCE_API:
    HF_TOKEN = st.secrets.get("HF_TOKEN", None)
    if not HF_TOKEN:
        st.warning("HF_TOKEN not found in Streamlit secrets. Set HF_TOKEN or set USE_HF_INFERENCE_API=False.")
    else:
        from huggingface_hub import InferenceClient
        hf_client = InferenceClient(token=HF_TOKEN)

else:
    qa_pipeline = get_qa_pipeline_local()

def retrieve_best_context(query, top_k=1):
    q_emb = embedder.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(q_emb, corpus_embeddings, top_k=top_k)[0]
    results = [bank.iloc[h['corpus_id']]['content'] for h in hits]
    return results[0] if results else ""

# Sidebar options
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of retrieved contexts (top_k)", 1, 5, 1)
    show_context = st.checkbox("Show retrieved context", value=False)
    st.markdown("---")
    st.markdown("Model hints:")
    st.markdown("- If the app fails to start on Streamlit Cloud, switch `LOCAL_MODEL` to a smaller model like `google/flan-t5-small` or use HF Inference API.")

# Chat UI
user_input = st.text_input("You:", "")
if st.button("Send") or user_input:
    if not user_input.strip():
        st.warning("Write a question first.")
    else:
        with st.spinner("Searching for best context..."):
            context = retrieve_best_context(user_input, top_k=top_k)

        if not context:
            st.info("Sorry, I couldn't find a relevant FAQ to answer your question.")
        else:
            if show_context:
                st.markdown("**Retrieved context:**")
                st.code(context)

            prompt = f"""
You are an expert Finance Chatbot working for a bank. Your job is to answer the customer's question professionally and clearly.

• You know banking products (loans, credit cards, accounts, deposits).
• You analyze customer queries using the provided context and answer concisely.
• If you don’t know the answer, say: "Sorry, I don't know."

Context: {context}
Customer Question: {user_input}
Answer:
"""

            # Use HuggingFace Inference API (remote) if requested
            if USE_HF_INFERENCE_API and st.secrets.get("HF_TOKEN", None):
                # remote inference
                inputs = {"inputs": prompt, "parameters": {"max_new_tokens": 200, "temperature": 0.3}}
                try:
                    out = hf_client.text_generation(model=HF_MODEL, **inputs)
                    # The structure may vary; convert to str safely
                    generated = out if isinstance(out, str) else str(out)
                    st.markdown(f"**Chatbot:** {generated}")
                except Exception as e:
                    st.error(f"HF inference error: {e}")
            else:
                # local pipeline inference
                try:
                    generated = qa_pipeline(prompt, max_new_tokens=200, do_sample=False, temperature=0.3)[0]["generated_text"]
                    st.markdown(f"**Chatbot:** {generated}")
                except Exception as e:
                    st.error(f"Inference failed: {e}. Consider switching to a smaller model or HF Inference API.")
