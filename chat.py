# --- 1. Database Fix (Must be at the very top) ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# --- 2. Imports ---
import streamlit as st
import pandas as pd
from langchain_community.llms import huggingface_endpoint
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
import os

# --- 3. Setup & Configuration ---
st.set_page_config(page_title="Banque Masr AI Assistant", page_icon="🏦", layout="centered")
st.title("🏦 Banque Masr Intelligent Assistant")

# Constants
REPO_ID = "HuggingFaceH4/zephyr-7b-beta"
DATA_PATH = "data/BankFAQs.csv" 

# Secrets Handling
if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
else:
    # Fallback for local testing if secrets aren't set
    api_key = st.sidebar.text_input("Enter Hugging Face Token", type="password")
    if api_key:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
    else:
        st.warning("Please enter your Hugging Face API Token in the sidebar or Streamlit Secrets.")
        st.stop()

# --- 4. Cached Resource Loading ---

@st.cache_resource
def load_data_and_vectordb():
    if not os.path.exists(DATA_PATH):
        st.error(f"File not found: {DATA_PATH}. Please check your GitHub folder structure.")
        return None

    bank = pd.read_csv(DATA_PATH)
    bank["content"] = bank.apply(lambda row: f"Question: {row['Question']}\nAnswer: {row['Answer']}", axis=1)
    
    documents = []
    for _, row in bank.iterrows():
        documents.append(Document(page_content=row["content"], metadata={"class": row["Class"]}))

    hg_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=hg_embeddings,
        collection_name="chatbot_BankMasr"
    )
    return vector_db

@st.cache_resource
def load_llm():
    # This is where your error was: HuggingFaceEndpoint must be imported above
    llm = HuggingFaceEndpoint(
        repo_id=REPO_ID,
        task="text-generation",
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        repetition_penalty=1.1
    )
    return llm

# --- 5. App Logic ---

with st.spinner("Initializing AI Brain..."):
    vector_db = load_data_and_vectordb()
    llm = load_llm()

if vector_db is None or llm is None:
    st.stop()

# Prompt Template
template = """<|system|>
You are a helpful and intelligent Finance QNA Expert for Banque Masr. 
Use the following context to answer the user's question accurately. 
If the answer is not in the context, say "Sorry, I don't know that information." and do not make up facts.
</s>
<|user|>
Context: {context}

Question: {question}
</s>
<|assistant|>
"""
PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

retriever = vector_db.as_retriever(search_kwargs={"k": 3})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT}
)

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome to Banque Masr! How can I help you today?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about loans, cards, or accounts..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = qa_chain.invoke(prompt)
                result = response['result']
                # Cleanup tokens if they appear in output
                if "<|assistant|>" in result:
                    result = result.split("<|assistant|>")[-1].strip()
                st.markdown(result)
                st.session_state.messages.append({"role": "assistant", "content": result})
            except Exception as e:
                st.error(f"Error: {str(e)}")
