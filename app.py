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
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import Document
from shiny import App, ui, render, reactive  # ✅ Shiny for Python
####################################################################################
###inititialize and define functions. 
####Chatgpt 4, and openai embeddings with ChormaDB is used
####################################################################################

### Set OpenAI API Key is set
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API Key is missing! Set OPENAI_API_KEY as an environment variable.")

# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

## writable temporary directory for ChromaDB
CHROMA_DB_DIR = tempfile.mkdtemp()

## reset ChromaDB as sometimes it is still holding on to old uploads
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
    print(f"ChromaDB reset. Storage location: {CHROMA_DB_DIR}")

# Initialize ChromaDB
reset_chromadb()

# LLM for Question-Answering (with max_tokens limit) set limits to keep the cost down
llm = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=OPENAI_API_KEY, max_tokens=500)

# Optimized Retriever (Fetch top 6 relevant chunks instead of 10). We are loosing some information but we wont hit the token limit
retriever = chroma_db.as_retriever(search_kwargs={"k": 6})

# QA Chain with Retriever llm and retreiver are defined above
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Function to count tokens before sending to GPT-4. This is added so we dont use up or reach limit when on large file is uploaded
def count_tokens(text):
    encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4's tokenizer
    return len(encoding.encode(text))

# ✅ Optimized PDF Processing
def process_pdf(file_path):
    """Extracts text from PDF, splits it into chunks, and indexes it in ChromaDB."""
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        # Text Splitting (400 chars + 50 overlap for better context)
        ####text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50, separators=["\n\n", "\n", " ", ""])
        docs = text_splitter.split_documents(pages)

        # Add text-based chunks to ChromaDB
        chroma_db.add_documents(docs)
        print(f"✅ Indexed {len(docs)} text chunks into ChromaDB.")

        return docs

    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        return []
#############################################################################################################
###start of ui layout
##############################################################################################################
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
            # ✅ Retrieve relevant documents
            retrieved_docs = retriever.get_relevant_documents(query)
            combined_text = " ".join([doc.page_content for doc in retrieved_docs])  # Combine retrieved chunks

            # ✅ Check token count before sending to GPT-4
            total_tokens = count_tokens(query + combined_text)
            print(f"🔢 Total tokens: {total_tokens}")

            if total_tokens > 7500:  # Keep a buffer below 8192
                answer_text.set("⚠️ Query is too long. Try asking a more specific question.")
                return

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
