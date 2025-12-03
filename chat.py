# --- 1. Database Fix (Must be at the very top) ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# --- 2. Imports ---
import streamlit as st
import pandas as pd
import os
import shutil

# Standard LangChain Imports (No 'classic' or 'streamlit_chat' needed)
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# --- 3. Setup & Configuration ---
st.set_page_config(page_title="Banque Masr AI Assistant", page_icon="🏦", layout="centered")
st.title("🏦 Banque Masr Conversational Assistant")

# Constants
# We use Flan-T5 because it is stable on the free API and good at following instructions
REPO_ID = "google/flan-t5-large"
CHROMA_PATH = "./chroma_db_data"

# --- 4. Secrets Handling ---
if "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
else:
    api_key = st.sidebar.text_input("Enter Hugging Face Token", type="password")
    if api_key:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
    else:
        st.warning("Please enter your Hugging Face API Token in the sidebar to proceed.")
        st.stop()

# --- 5. Data & Resources ---

@st.cache_resource
def load_data_and_vectordb():
    # 1. Locate the CSV
    possible_paths = ["data/BankFAQs.csv", "BankFAQs.csv"]
    file_path = None
    for path in possible_paths:
        if os.path.exists(path):
            file_path = path
            break
            
    if not file_path:
        st.error("❌ File not found! Please upload 'BankFAQs.csv' to GitHub.")
        return None

    try:
        # 2. Reset DB to prevent corruption errors
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)

        # 3. Load and Process Data
        bank = pd.read_csv(file_path)
        bank["content"] = bank.apply(lambda row: f"Question: {row['Question']}\nAnswer: {row['Answer']}", axis=1)
        
        documents = []
        for _, row in bank.iterrows():
            documents.append(Document(page_content=row["content"], metadata={"class": row["Class"]}))

        # 4. Create Embeddings & DB
        hg_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_db = Chroma.from_documents(
            documents=documents,
            embedding=hg_embeddings,
            persist_directory=CHROMA_PATH,
            collection_name="chatbot_BankMasr"
        )
        return vector_db
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def load_llm():
    try:
        # Load Flan-T5 (Fixes the "Task not supported" error)
        llm = HuggingFaceEndpoint(
            repo_id=REPO_ID,
            task="text2text-generation",
            max_new_tokens=512,
            do_sample=False,
            temperature=0.1
        )
        return llm
    except Exception as e:
        st.error(f"Error loading Model: {e}")
        return None

# --- 6. Initialization ---

with st.spinner("Initializing AI Brain..."):
    vector_db = load_data_and_vectordb()
    llm = load_llm()

if vector_db is None or llm is None:
    st.stop()

# Setup Memory (This remembers your previous questions)
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer' 
    )

# Setup Chain
# We use ConversationalRetrievalChain to handle history automatically
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
    memory=st.session_state.memory,
    return_source_documents=False,
    verbose=False
)

# --- 7. Chat Interface (Native) ---

# Initialize chat history for UI
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome to Banque Masr! How can I help you today?"}]

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle User Input
if prompt := st.chat_input("Ask about loans, cards, or accounts..."):
    # 1. Add user message to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    # 2. Generate Answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # The chain automatically handles history via st.session_state.memory
                response = chain.invoke({"question": prompt})
                result = response["answer"]
                
                st.markdown(result)
                st.session_state.messages.append({"role": "assistant", "content": result})
            except Exception as e:
                st.error(f"Error: {str(e)}")
