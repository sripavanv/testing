import os
import chromadb
import tempfile
import tiktoken
import pandas as pd
import PyPDF2
from PIL import Image, UnidentifiedImageError
import io
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import Document
from shiny import App, ui, render, reactive
import fitz  # PyMuPDF for PDF image extraction

####################################################################################
### Initialize and define functions. 
####################################################################################

# Set OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API Key is missing! Set OPENAI_API_KEY as an environment variable.")

# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Writable temporary directory for ChromaDB
CHROMA_DB_DIR = tempfile.mkdtemp()

# Reset ChromaDB
def reset_chromadb():
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
    print(f"ChromaDB reset. Storage location: {CHROMA_DB_DIR}")

# Initialize ChromaDB
reset_chromadb()

# LLM Model Setup
llm = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=OPENAI_API_KEY, max_tokens=500)

# Function to count tokens
def count_tokens(text):
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

# ✅ Optimized PDF Processing with Page Numbers
def process_pdf(file_path):
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        docs = text_splitter.split_documents(pages)

        # ✅ Store page numbers in metadata
        for i, doc in enumerate(docs):
            doc.metadata["page"] = pages[i // len(pages)].metadata["page"]  # Assign page number

        chroma_db.add_documents(docs)
        print(f"✅ Indexed {len(docs)} text chunks into ChromaDB.")

        return docs
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        return []

# ✅ Extract Image of Section from PDF
def extract_section_image(file_path, page_number):
    try:
        doc = fitz.open(file_path)  # Open PDF
        page = doc[page_number - 1]  # Get Page (1-based index)

        # Convert full page to an image
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        return img
    except Exception as e:
        print(f"❌ Error extracting image: {e}")
        return None

#############################################################################################################
### UI Layout
##############################################################################################################
app_ui = ui.page_fluid(
    ui.h2("📄 AI-Powered PDF Analyzer"),
    ui.input_file("file", "Upload PDF Document", multiple=False, accept=[".pdf"]),
    ui.input_text("query", "Ask a question about the document"),
    ui.input_action_button("ask", "Ask AI"),
    ui.output_text("result_text"),
    ui.output_image("section_image")  # ✅ Display section image
)

# ✅ Server Logic
def server(input, output, session):
    uploaded_file_path = reactive.value("")
    answer_text = reactive.value("")
    retrieved_page = reactive.value(None)

    @reactive.effect
    @reactive.event(input.file)
    def handle_file_upload():
        file_info = input.file()
        if not file_info:
            return

        file_path = file_info[0]["datapath"]
        uploaded_file_path.set(file_path)

        reset_chromadb()
        docs = process_pdf(file_path)

        if docs:
            print("✅ PDF uploaded and processed successfully.")
        else:
            print("❌ Failed to process PDF.")

    @reactive.effect
    @reactive.event(input.ask)
    def generate_response():
        query = input.query()
        if not query:
            answer_text.set("Please enter a valid question.")
            return

        if chroma_db._collection.count() == 0:
            answer_text.set("No PDF uploaded. Please upload a file before asking questions.")
            return

        print(f"📝 Query received: {query}")

        try:
            retrieved_docs = chroma_db.as_retriever().get_relevant_documents(query)
            if not retrieved_docs:
                answer_text.set("No relevant information found.")
                return

            retrieved_text = " ".join([doc.page_content for doc in retrieved_docs])
            retrieved_page.set(retrieved_docs[0].metadata["page"])  # Store first relevant page

            total_tokens = count_tokens(query + retrieved_text)
            if total_tokens > 7500:
                answer_text.set("⚠️ Query too long. Try asking a more specific question.")
                return

            result = RetrievalQA.from_chain_type(llm=llm, retriever=chroma_db.as_retriever()).invoke(query)
            answer_text.set(result["result"] if "result" in result else str(result))

        except Exception as e:
            answer_text.set("Error retrieving response.")
            print(f"❌ Error: {e}")

    @render.text
    def result_text():
        return answer_text.get()

    @render.image
    def section_image():
        file_path = uploaded_file_path.get()
        page_number = retrieved_page.get()
        if not file_path or not page_number:
            return None

        img = extract_section_image(file_path, page_number)
        if img:
            img_path = os.path.join(tempfile.gettempdir(), "section_image.png")
            img.save(img_path)
            return img_path
        return None

# ✅ Run Shiny App
app = App(app_ui, server)
