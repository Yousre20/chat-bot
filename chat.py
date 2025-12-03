# --- 1. Database Fix (Must be at the very top) ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# --- 2. Imports ---
import streamlit as st
import os
import shutil # NEW: Needed to delete old database files
import pandas as pd

try:
    from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import PromptTemplate
    from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_core.documents import Document
    import os



except ImportError as e:
    st.error(f"❌ Library Missing: {e}. Please ensure 'requirements.txt' contains 'langchain', 'langchain-huggingface', and 'langchain-chroma'.")
    st.stop()

# --- 3. Setup & Configuration ---
st.set_page_config(page_title="Banque Masr AI Assistant", page_icon="🏦", layout="centered")
st.title("🏦 Banque Masr Intelligent Assistant")

# Constants
REPO_ID = "google/flan-t5-large"
CHROMA_PATH = "./chroma_db_data" # Specific folder for the DB

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

# --- 5. Smart Resource Loading ---

@st.cache_resource
def load_data_and_vectordb():
    # Smart Path Checking
    possible_paths = ["data/BankFAQs.csv", "BankFAQs.csv"]
    file_path = None
    
    for path in possible_paths:
        if os.path.exists(path):
            file_path = path
            break
            
    if not file_path:
        st.error("❌ File not found! Please upload 'BankFAQs.csv' to your GitHub repository.")
        return None

    try:
        # NEW: Self-Cleaning! 
        # Delete the old database directory if it exists to fix "no such table" errors
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)

        bank = pd.read_csv(file_path)
        bank["content"] = bank.apply(lambda row: f"Question: {row['Question']}\nAnswer: {row['Answer']}", axis=1)
        
        documents = []
        for _, row in bank.iterrows():
            documents.append(Document(page_content=row["content"], metadata={"class": row["Class"]}))

        hg_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Create fresh DB
        vector_db = Chroma.from_documents(
            documents=documents,
            embedding=hg_embeddings,
            persist_directory=CHROMA_PATH, # Force it into our clean folder
            collection_name="chatbot_BankMasr"
        )
        return vector_db
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def load_llm():
    try:
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

# --- 6. App Logic ---

with st.spinner("Initializing AI Brain..."):
    vector_db = load_data_and_vectordb()
    llm = load_llm()

if vector_db is None or llm is None:
    st.stop()

template = """Use the following pieces of context to answer the question at the end. 
If the answer is not in the context, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {input}

Answer:"""

PROMPT = PromptTemplate.from_template(template)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

question_answer_chain = create_stuff_documents_chain(llm, PROMPT)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# --- Chat Interface ---
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
                response = rag_chain.invoke({"input": prompt})
                result = response["answer"]
                st.markdown(result)
                st.session_state.messages.append({"role": "assistant", "content": result})
            except Exception as e:
                st.error(f"Error: {str(e)}")


