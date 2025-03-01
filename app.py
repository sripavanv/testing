import os
import chromadb
import tempfile
import pandas as pd
import PyPDF2
from PIL import Image, UnidentifiedImageError
import io
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import Document
import shutil
import time
from shiny import App, ui, render, reactive  # ✅ Shiny for Python

# ✅ Ensure OpenAI API Key is set
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌ OpenAI API Key is missing! Set OPENAI_API_KEY as an environment variable.")

# ✅ Use a writable temporary directory for ChromaDB
CHROMA_DB_DIR = tempfile.mkdtemp()

# ✅ Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# ✅ Function to reset ChromaDB efficiently
def reset_chromadb():
    """Clears ChromaDB and reinitializes it with a clean slate."""
    global chroma_db

    try:
        if 'chroma_db' in globals() and chroma_db is not None:
            chroma_db.delete(ids=None)
            chroma_db.persist()
            del chroma_db
            print("🗑️ ChromaDB records cleared.")
    except Exception as e:
        print(f"⚠️ Warning: Could not clear ChromaDB properly: {e}")

    chroma_db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    print(f"✅ ChromaDB reset. Storage location: {CHROMA_DB_DIR}")

# ✅ Initialize ChromaDB
reset_chromadb()

# ✅ LLM for Question-Answering (with max_tokens limit)
llm = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=OPENAI_API_KEY, max_tokens=500)

# ✅ Optimized Retriever (Fetch top 10 relevant chunks)
retriever = chroma_db.as_retriever(search_kwargs={"k": 10})

# ✅ QA Chain with Retriever
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# ✅ Optimized PDF Processing
def process_pdf(file_path):
    """Extracts text from PDF, splits it into chunks, and indexes it in ChromaDB."""
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        # ✅ Smart Text Splitting (600 chars + 100 overlap for better context)
        text_splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        docs = text_splitter.split_documents(pages)

        # ✅ Add text-based chunks to ChromaDB
        chroma_db.add_documents(docs)
        print(f"✅ Indexed {len(docs)} text chunks into ChromaDB.")

        return docs

    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        return []

# ✅ UI Layout
app_ui = ui.page_fluid(
    ui.h2("📄 AI-Powered PDF Analyzer"),
    ui.input_file("file", "Upload PDF Document", multiple=False, accept=[".pdf"]),
    ui.input_text("query", "Ask a question about the document"),
    ui.input_action_button("ask", "Ask AI"),
    ui.output_text("result_text")
)

# ✅ Server Logic
def server(input, output, session):
    """Handles file uploads and AI interactions"""

    @reactive.effect
    @reactive.event(input.file)
    def handle_file_upload():
        """Processes uploaded PDF file and indexes it in ChromaDB."""
        file_info = input.file()
        if not file_info:
            return

        file_path = file_info[0]["datapath"]

        # ✅ Only reset ChromaDB if a new file is uploaded
        reset_chromadb()

        docs = process_pdf(file_path)  # ✅ Process and store new data

        if docs:
            print("✅ PDF uploaded and processed successfully.")
        else:
            print("❌ Failed to process PDF.")

    answer_text = reactive.value("")

    @reactive.effect
    @reactive.event(input.ask)
    def generate_response():
        """Handles GPT-4 response based on indexed PDF content."""
        query = input.query()
        if not query:
            print("❌ No query provided.")
            answer_text.set("Please enter a valid question.")
            return

        # ✅ Ensure ChromaDB has indexed data before querying
        if chroma_db._collection.count() == 0:
            print("❌ No PDF uploaded. Please upload a file first.")
            answer_text.set("No PDF uploaded. Please upload a file before asking questions.")
            return

        print(f"📝 Query received: {query}")

        try:
            # ✅ Retrieve context and query GPT-4
            result = qa_chain.invoke(query)
            result_text = result["result"] if isinstance(result, dict) and "result" in result else str(result)
            answer_text.set(result_text if result_text else "No relevant information found.")

        except Exception as e:
            print(f"❌ Error retrieving answer: {e}")
            answer_text.set("Error retrieving response.")

    @render.text
    def result_text():
        return answer_text.get() if answer_text.get() else "No response yet."

# ✅ Run Shiny App
app = App(app_ui, server)
