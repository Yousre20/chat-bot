# --- 1. Database Fix (Must be at the very top) ---
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
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
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

# --- 6. Load Resources ---
@st.cache_resource
def load_data_and_db():
    # File check
    files = ["data/BankFAQs.csv", "BankFAQs.csv"]
    file_path = next((f for f in files if os.path.exists(f)), None)
    
    if not file_path:
        st.error("❌ 'BankFAQs.csv' not found.")
        return None

    # Reset DB
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Process Data
    bank = pd.read_csv(file_path)
    bank["content"] = bank.apply(lambda row: f"Q: {row['Question']}\nA: {row['Answer']}", axis=1)
    docs = [Document(page_content=row["content"]) for _, row in bank.iterrows()]

    # Vector DB
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_PATH)

@st.cache_resource
def load_llm():
    return HuggingFaceEndpoint(
        repo_id=REPO_ID, 
        task="text2text-generation", 
        temperature=0.1,
        max_new_tokens=512
    )

# --- 7. App Logic ---
with st.spinner("Starting AI..."):
    vector_db = load_data_and_db()
    llm = load_llm()

if not vector_db or not llm:
    st.stop()

# --- 8. The New Chain (Fixes Import Errors) ---

# 1. Contextualize Question Prompt (For History)
contextualize_q_system_prompt = """Given a chat history and the latest user question 
which might reference context in the chat history, formulate a standalone question 
which can be understood without the chat history. Do NOT answer the question, 
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# 2. History Aware Retriever
history_aware_retriever = create_history_aware_retriever(
    llm, vector_db.as_retriever(search_kwargs={"k": 2}), contextualize_q_prompt
)

# 3. Answer Prompt
qa_system_prompt = """You are an assistant for Banque Masr. Use the following pieces of 
retrieved context to answer the question. If you don't know the answer, say that you 
don't know. Keep the answer concise.

{context}"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# 4. Final Chain
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# --- 9. Chat UI ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    st.chat_message(role).write(msg.content)

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    
    with st.chat_message("assistant"):
        response = rag_chain.invoke({
            "input": prompt,
            "chat_history": st.session_state.chat_history
        })
        st.write(response["answer"])
        
    # Update History
    st.session_state.chat_history.append(HumanMessage(content=prompt))
    st.session_state.chat_history.append(AIMessage(content=response["answer"]))
