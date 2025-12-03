# --- 1. Database Fix (Must be at the very top) ---
# Required for running Chroma DB on some environments (like Streamlit Cloud)
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass 

# --- 2. Imports ---
import streamlit as st
# Imports for conversational components and UI
from streamlit_chat import message
# FIXED: Use standard LangChain imports
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
# Imports for RAG and LLM setup
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate # Added for custom RAG prompting
import os
import pandas as pd 

# --- 3. Banque Masr Static Data (Replacing BankFAQs.csv) ---
BANK_FAQS = [
    {"question": "What are the requirements for opening a new savings account?", "answer": "You need a valid national ID, a recent utility bill, and an initial deposit of 1000 EGP.", "class": "Accounts"},
    {"question": "What is the maximum duration for a personal loan?", "answer": "The maximum duration is 7 years, or 84 months, subject to credit score approval.", "class": "Loans"},
    {"question": "How do I report a lost or stolen credit card?", "answer": "Immediately call our 24/7 hotline at 19666. Your card will be instantly blocked. This is a critical security measure.", "class": "Cards"},
    {"question": "Can I apply for a mortgage if I am self-employed?", "answer": "Yes, provided you can show consistent income statements for the past two years and provide business registration documents.", "class": "Loans"},
    {"question": "What are the monthly maintenance fees for the Platinum account?", "answer": "The monthly maintenance fee is 50 EGP, which is waived if the minimum balance of 20,000 EGP is maintained for the entire month.", "class": "Accounts"},
    {"question": "Do you offer car loans?", "answer": "Yes, car loans are available for both new and used vehicles, with repayment terms up to 5 years.", "class": "Loans"},
]
# Constants
REPO_ID = "mistralai/Mistral-7B-Instruct-v0.2"

# Prompt Template for Conversational Chain (using Mistral formatting)
CUSTOM_TEMPLATE = """<|system|>
You are a helpful and intelligent Finance QNA Expert for Banque Masr. 
Use the following context to answer the user's question accurately. 
If the answer is not in the context, say "Sorry, I don't know that information." and do not make up facts.

Chat History:
{chat_history}
</s>
<|user|>
Context: {context}

Question: {question}
</s>
<|assistant|>
"""
PROMPT = PromptTemplate(
    input_variables=["context", "question", "chat_history"], 
    template=CUSTOM_TEMPLATE
)


# --- 4. Setup & Configuration ---
st.set_page_config(page_title="Banque Masr AI Assistant", page_icon="🏦", layout="centered")
st.title("🏦 Banque Masr Conversational Assistant :books:")

# Secrets Handling
if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
    # FIXED: Use st.sidebar for text input
    api_key = st.sidebar.text_input("Enter Hugging Face API Token", type="password") 
    if api_key:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
    else:
        st.warning("Please enter your Hugging Face API Token in the sidebar.")
        st.stop()


# --- 5. Cached Resource Loading ---

@st.cache_resource
def load_data_and_vectordb():
    # Process embedded data into LangChain Documents
    documents = []
    for faq in BANK_FAQS:
        content = f"Question: {faq['question']}\nAnswer: {faq['answer']}"
        documents.append(Document(page_content=content, metadata={"class": faq["class"]}))

    # Create embeddings
    hg_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create vector store (Chroma)
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=hg_embeddings,
        collection_name="chatbot_BankMasr"
    )
    return vector_db

@st.cache_resource
def load_llm():
    # Initialize HuggingFace Endpoint LLM (using your Mistral model)
    llm = HuggingFaceEndpoint(
        repo_id=REPO_ID,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        repetition_penalty=1.1,
        task="text-generation" 
    )
    return llm


# --- 6. Conversational Chain Functions ---

def initialize_session_state():
    # Initializes the state variables needed by streamlit_chat and the chain
    if 'history' not in st.session_state:
        st.session_state['history'] = [] 
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Welcome to Banque Masr! How can I help you today?"]
    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

def create_conversational_chain(vector_store, llm):
    # Setup conversation memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Create the ConversationalRetrievalChain (replaces RetrievalQA)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        chain_type='stuff',
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        # Pass the custom prompt to ensure persona and history are used
        combine_docs_chain_kwargs={"prompt": PROMPT} 
    )
    return chain

def conversation_chat(query, chain):
    # This function invokes the chain, which manages history internally
    result = chain({"question": query}) 
    answer = result["answer"]

    # Clean up Mistral-specific tokens
    if "<|assistant|>" in answer:
        answer = answer.split("<|assistant|>")[-1].strip()
        
    return answer

def display_chat_history(chain):
    # Containers to manage layout
    reply_container = st.container()
    container = st.container()

    with container:
        # Use a standard chat input instead of st.form for cleaner UX
        user_input = st.chat_input("Ask about loans, cards, or accounts...")

        if user_input:
            with st.spinner('Searching FAQs...'):
                output = conversation_chat(user_input, chain)

            # Update session state for the UI
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    # Display chat history using streamlit_chat.message
    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                # Skip the initial user greeting that doesn't correspond to an input
                if i < len(st.session_state["past"]):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")


# --- 7. Main Execution ---

def main():
    # Initialize session state (past, generated, history)
    initialize_session_state()

    st.sidebar.subheader("RAG Model Status")
    
    # Load resources
    try:
        # CORRECTED: Uses st.spinner globally and st.sidebar.info for status updates
        with st.spinner("Loading RAG resources..."):
            st.sidebar.info("1. Preparing knowledge base (Chroma DB)...")
            vector_db = load_data_and_vectordb()
            st.sidebar.success("1. Knowledge base ready!")
            
            st.sidebar.info("2. Loading LLM (Mistral)...")
            llm = load_llm()
            st.sidebar.success("2. LLM loaded!")
            
    except Exception as e:
        # If an error still occurs, print the full error
        st.error(f"Failed to initialize RAG components: {e}")
        st.stop()
    
    st.sidebar.success("All systems ready!")

    # Create the Conversational Chain object
    chain = create_conversational_chain(vector_db, llm)
    
    # Display the conversational UI
    display_chat_history(chain)

if __name__ == "__main__":
    main()
