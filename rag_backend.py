# rag_backend.py
# All RAG logic: load URLs, embed, store, and create conversational chain
import os

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import bs4

# ─── CONFIG ───────────────────────────────────────────────────────────────────

from dotenv import load_dotenv

# This looks for the .env file and loads the variables
load_dotenv()

# Now fetch the key safely
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY! Please check your .env file.")


URLS_TO_CRAWL = [
    "https://www.rvitm.edu.in/",
    "https://www.rvitm.edu.in/computer-science-engineering/",
    "https://www.rvitm.edu.in/computer-science-and-engineering-ai-ml/",
    "https://www.rvitm.edu.in/admission/"
    "https://www.rvitm.edu.in/electronics-and-communication-engineering/"
]

# ─── STEP 1: LOAD WEB PAGES ───────────────────────────────────────────────────

def load_documents():
    loader = WebBaseLoader(
        web_paths=URLS_TO_CRAWL,
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "section"]
            )
        )
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} pages")
    return docs

# ─── STEP 2: SPLIT INTO CHUNKS ────────────────────────────────────────────────

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=128)
    split_docs = splitter.split_documents(docs)
    print(f"Split into {len(split_docs)} chunks")
    return split_docs

# ─── STEP 3: EMBED AND STORE IN QDRANT ───────────────────────────────────────

def create_vectorstore(split_docs):
    embeddings = FastEmbedEmbeddings()
    vectorstore = QdrantVectorStore.from_documents(
        split_docs,
        embedding=embeddings,
        path="./collage2_db",
        collection_name="demo",
    )
    print("Vectorstore created")
    return vectorstore

# ─── STEP 4: BUILD CONVERSATIONAL RAG CHAIN ──────────────────────────────────

def build_chain(retriever):
    llm = ChatGroq(
        model_name='llama-3.1-8b-instant',
        temperature=0,
        api_key=GROQ_API_KEY
    )

    # System prompt for answering
    system_prompt = (
    "You are a helpful assistant for RV Institute of Technology and Management (RVITM), Bengaluru. "
    "You only answer questions about RVITM — not about any other RV group institutions like RVCE or others. "
    "Use the provided context to answer the user's question. "
    "If you don't find the answer in the context, respond with 'Sorry! I don't know'. "
    "Format the response for easy reading. "
    "Use three sentences maximum and keep the answer concise.\n\n"
    "{context}"
)

    # Prompt to rephrase question based on chat history
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the chat history and the latest user question, "
         "rephrase it as a standalone question. "
         "Do NOT answer it, just rephrase if needed."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # Prompt for final answer
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # Build chain
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Session store
    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversational_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_chain

# ─── MAIN: INITIALISE EVERYTHING ─────────────────────────────────────────────

def init_rag_chain():
    docs = load_documents()
    split_docs = split_documents(docs)
    vectorstore = create_vectorstore(split_docs)
    retriever = vectorstore.as_retriever(search_kwargs={'k': 5})
    chain = build_chain(retriever)
    print("RAG chain ready!")
    return chain