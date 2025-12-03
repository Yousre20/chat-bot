import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
import os

# --- Page Config ---
st.set_page_config(page_title="Banque Masr AI Assistant", page_icon="🏦", layout="centered")
st.title("🏦 Banque Masr Intelligent Assistant")

# --- Constants ---
MODEL_ID = 'HuggingFaceH4/zephyr-7b-beta'
DATA_PATH = "data/BankFAQs.csv" # Ensure your CSV is in a 'data' folder

# --- 1. Load Resources (Cached) ---

@st.cache_resource
def load_data_and_vectordb():
    """Loads CSV, processes it, and sets up ChromaDB."""
    if not os.path.exists(DATA_PATH):
        st.error(f"File not found: {DATA_PATH}. Please upload the CSV.")
        return None

    # Load Data
    bank = pd.read_csv(DATA_PATH)
    bank["content"] = bank.apply(lambda row: f"Question: {row['Question']}\nAnswer: {row['Answer']}", axis=1)
    
    # Create Documents
    documents = []
    for _, row in bank.iterrows():
        documents.append(Document(page_content=row["content"], metadata={"class": row["Class"]}))

    # Embeddings
    hg_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Vector DB (In-memory for session or persistent)
    # Using a persistent directory so it doesn't rebuild unnecessarily
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=hg_embeddings,
        collection_name="chatbot_BankMasr",
        persist_directory="./chroma_db_streamlit"
    )
    return vector_db

@st.cache_resource
def load_llm():
    """Loads the Zephyr LLM with 4-bit Quantization."""
    
    # Quantization Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Load Model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map='auto'
    )

    # Create Pipeline
    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": torch.float16},
        max_new_tokens=500,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        return_full_text=False # Important for LangChain
    )

    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    return llm

# --- 2. Initialize Application ---

with st.spinner("Initializing AI Brain... (This depends on your GPU)"):
    vector_db = load_data_and_vectordb()
    llm = load_llm()

if vector_db is None or llm is None:
    st.stop()

# --- 3. Setup QA Chain ---

template = """
You are a Finance QNA Expert for Banque Masr. Analyze the Query and Respond to the Customer with a suitable and helpful answer based ONLY on the context provided below.
If you don't know the answer based on the context, just say "Sorry, I don't know." do not make up facts.

Context: {context}

Question: {question}

Answer:
"""
PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

retriever = vector_db.as_retriever(search_kwargs={"k": 2})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT}
)

# --- 4. Chat Interface ---

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome to Banque Masr! How can I help you today?"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask about loans, accounts, or cards..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Run the chain
                response = qa_chain.invoke(prompt)
                result = response['result']
                st.markdown(result)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": result})
            except Exception as e:
                st.error(f"An error occurred: {e}")
