# --- 1. Database Fix (Must be at the very top) ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# --- 2. Imports ---
import streamlit as st
import pandas as pd
import os
import shutil

# Modern Imports (These WORK with the requirements above)
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
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

# --- 8. The Modern Chain ---

# A. Search Query Generator (Rewrites user question based on history)
context_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given a chat history and the latest user question, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
history_retriever = create_history_aware_retriever(llm, vector_db.as_retriever(search_kwargs={"k": 2}), context_prompt)

# B. Question Answering Chain
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant for Banque Masr. Answer the question based ONLY on the following context:\n\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_retriever, question_answer_chain)

# --- 9. Chat UI ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    st.chat_message(role).write(msg.content)

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = rag_chain.invoke({
                    "input": prompt,
                    "chat_history": st.session_state.chat_history
                })
                output = response["answer"]
                st.write(output)
                
                # Save History
                st.session_state.chat_history.append(HumanMessage(content=prompt))
                st.session_state.chat_history.append(AIMessage(content=output))
            except Exception as e:
                st.error(f"Error: {e}")
