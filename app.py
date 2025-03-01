import os
import chromadb
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
from shiny import App, ui, render, reactive  # ✅ FIXED SHINY IMPORT

# ✅ Ensure API Key is set
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌ OpenAI API Key is missing! Set OPENAI_API_KEY as an environment variable.")

# ✅ Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# ✅ Function to completely reset ChromaDB
def reset_chromadb():
    """Completely deletes ChromaDB's database and resets everything to ensure fresh indexing."""
    global chroma_db

    try:
        # ✅ If ChromaDB exists, delete all records
        if 'chroma_db' in globals() and chroma_db is not None:
            chroma_db.delete(ids=None)  # ✅ Deletes all stored records
            chroma_db.persist()  # ✅ Ensure deletion is saved
            del chroma_db  # ✅ Delete reference to force reload
            print("🗑️ ChromaDB records successfully deleted.")
    except Exception as e:
        print(f"⚠️ Warning: Could not clear ChromaDB records properly: {e}")

    # ✅ Ensure ChromaDB directory is fully deleted to prevent cache issues
    for _ in range(3):  # Retry up to 3 times to avoid file lock issues
        try:
            if os.path.exists("./chroma_db"):
                shutil.rmtree("./chroma_db")  # ✅ Remove the database folder
                print("🗑️ ChromaDB directory deleted.")
            break
        except PermissionError:
            print("⚠️ ChromaDB is locked, retrying in 2 seconds...")
            time.sleep(2)  # Wait and retry

    # ✅ Reinitialize ChromaDB
    chroma_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    print("✅ ChromaDB fully reset and reinitialized.")

# ✅ Initialize ChromaDB
reset_chromadb()

# ✅ LLM for Question-Answering
llm = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=OPENAI_API_KEY)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=chroma_db.as_retriever())

# ✅ Function to process PDFs (Extracts text + images)
def process_pdf(file_path):
    """Extracts text and images from PDFs and stores them in ChromaDB."""
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(pages)

        # ✅ Store text-based docs
        chroma_db.add_documents(docs)
        print(f"✅ Processed {len(docs)} text chunks.")

        return docs

    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        return []

# ✅ UI Layout
app_ui = ui.page_fluid(
    ui.h2("📄 AI-Powered PDF Analyzer"),
    ui.input_file("file", "Upload PDF Document", multiple=False, accept=[".pdf"]),
    ui.input_text("query", "Ask a question about the document"),
    ui.input_action_button("ask", "Ask AI"),  # Button to trigger search
    ui.output_text("result_text")
)

# ✅ Server Logic
def server(input, output, session):
    """Handles user interactions and AI processing"""

    @reactive.effect
    @reactive.event(input.file)
    def handle_file_upload():
        """Processes uploaded PDF file and indexes it into ChromaDB."""
        file_info = input.file()
        if not file_info:
            return

        file_path = file_info[0]["datapath"]  # Get uploaded file path

        # ✅ Fully clear ChromaDB before processing a new file
        reset_chromadb()

        docs = process_pdf(file_path)  # Extract & store new data

        if docs:
            print("✅ PDF uploaded and processed.")
        else:
            print("❌ Failed to process PDF.")

    answer_text = reactive.value("")

    @reactive.effect
    @reactive.event(input.ask)
    def generate_response():
        """Handles GPT-4 response and retrieves relevant content."""
        query = input.query()
        if not query:
            print("❌ No query provided.")
            answer_text.set("Please enter a valid question.")
            return

        # ✅ Check if PDFs have been uploaded (if ChromaDB has data)
        if chroma_db._collection.count() == 0:
            print("❌ No PDF uploaded. Please upload a file first.")
            answer_text.set("No PDF uploaded. Please upload a file before asking questions.")
            return

        print(f"📝 Query received: {query}")

        try:
            # ✅ Otherwise, use GPT-4 for text-based queries
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