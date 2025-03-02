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
from shiny import App, ui, render, reactive  # ✅ Shiny for Python

####################################################################################
### Initialize and define functions. 
### ChatGPT-4, OpenAI embeddings, and ChromaDB are used.
####################################################################################

# ✅ Set OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API Key is missing! Set OPENAI_API_KEY as an environment variable.")

# ✅ Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# ✅ Writable temporary directory for ChromaDB
CHROMA_DB_DIR = tempfile.mkdtemp()

# ✅ Reset ChromaDB (but not on every upload)
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

# ✅ Initialize ChromaDB
reset_chromadb()

# ✅ LLM for Question-Answering (limits to keep costs down)
llm = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=OPENAI_API_KEY, max_tokens=500)

# ✅ Optimized Retriever (Fetches top 6 relevant chunks)
retriever = chroma_db.as_retriever(search_kwargs={"k": 6})

# ✅ QA Chain with Retriever
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# ✅ Function to count tokens before sending to GPT-4
def count_tokens(text):
    encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4's tokenizer
    return len(encoding.encode(text))

# ✅ Optimized PDF Processing for Multiple PDFs
def process_pdf(file_path):
    """Extracts text from PDFs, splits it into chunks, and indexes it in ChromaDB."""
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        file_name = os.path.basename(file_path)  # ✅ Extract filename
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50, separators=["\n\n", "\n", " ", ""])
        docs = text_splitter.split_documents(pages)

        # ✅ Attach filename as metadata to each chunk
        for doc in docs:
            doc.metadata["source"] = file_name  

        # ✅ Add text-based chunks to ChromaDB
        chroma_db.add_documents(docs)

        print(f"✅ Indexed {len(docs)} text chunks from {file_name}.")
        return docs
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return []

#############################################################################################################
### ✅ Start of UI Layout
##############################################################################################################
app_ui = ui.page_fluid(
    ui.h2("📄 AI-Powered Multi-PDF Analyzer"),
    ui.h6("RAG-based LLM. Prompts will impact retrieval quality."),
    ui.h6("Use document section titles for better results."),
    ui.h6("Avoid large files to reduce processing costs."),
    ui.h6("Technical papers (< 80 pages) are ideal test cases."),
    ui.h6("This version retrieves fewer chunks to keep costs low."),
    ui.h6("If responses seem incomplete, the enterprise version can scale."),
    ui.h6("⚠️ Known issue: Re-uploading after first upload may cause issues. Restart the app if needed."),
    ui.input_file("file", "Upload PDF Documents", multiple=True, accept=[".pdf"]),  # ✅ Supports multiple files
    ui.input_text("query", "Ask a question about the documents"),
    ui.input_action_button("ask", "Ask AI"),
    ui.output_text("result_text")
)

# ✅ Server Logic
def server(input, output, session):
    """Handles file uploads and AI interactions"""

    @reactive.effect
    @reactive.event(input.file)
    def handle_file_upload():
        """Processes multiple uploaded PDF files and indexes them in ChromaDB."""
        file_info = input.file()
        if not file_info:
            return

        new_docs = []
        for file in file_info:
            file_path = file["datapath"]
            docs = process_pdf(file_path)  # ✅ Process each PDF
            new_docs.extend(docs)  # ✅ Store all extracted documents

        if new_docs:
            print(f"✅ Indexed {len(new_docs)} chunks from {len(file_info)} PDFs into ChromaDB.")
        else:
            print("❌ No valid PDFs processed.")

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
            answer_text.set("No PDFs uploaded. Please upload files before asking questions.")
            return

        print(f"📝 Query received: {query}")

        try:
            # ✅ Retrieve relevant documents
            retrieved_docs = retriever.get_relevant_documents(query)

            # ✅ Combine retrieved content with document source names
            combined_text = "\n\n".join(
                [f"📄 Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}" for doc in retrieved_docs]
            )

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
