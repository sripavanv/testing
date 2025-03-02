import os
import chromadb
import tempfile
import tiktoken
import PyPDF2
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from shiny import App, ui, render, reactive  

####################################################################################
### Initialization
####################################################################################

# ✅ OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API Key is missing! Set OPENAI_API_KEY as an environment variable.")

# ✅ Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# ✅ Temporary directory for ChromaDB
CHROMA_DB_DIR = tempfile.mkdtemp()

# ✅ Reset ChromaDB 
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

# ✅ Initialize ChromaDB
reset_chromadb()

# ✅ LLM for Q&A
llm = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=OPENAI_API_KEY, max_tokens=700)

# ✅ Optimized Retriever
retriever = chroma_db.as_retriever(search_kwargs={"k": 15})  # 🔥 Retrieve more chunks to get full sections

# ✅ QA Chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# ✅ Token Counter
def count_tokens(text):
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

# ✅ Extract PDF Title Function
def extract_pdf_title(file_path):
    """Extracts the title from the PDF metadata. Defaults to filename if unavailable."""
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            metadata = reader.metadata
            if metadata and "/Title" in metadata:
                return metadata["/Title"]
    except Exception as e:
        print(f"⚠️ Could not extract title from {file_path}: {e}")
    
    return os.path.basename(file_path)  # Default to filename if no title found

# ✅ Process PDFs & Preserve Structure
def process_pdf(file_path):
    """Extracts text from PDFs, splits it into chunks, and indexes it in ChromaDB while preserving section order."""
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        pdf_title = extract_pdf_title(file_path)  # ✅ Get the title
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""])
        docs = text_splitter.split_documents(pages)

        # ✅ Attach metadata (title, source, chunk index, and section headers)
        for i, doc in enumerate(docs):
            doc.metadata["title"] = pdf_title  
            doc.metadata["source"] = os.path.basename(file_path)
            doc.metadata["chunk_index"] = i  # ✅ Preserve order

            # ✅ Extract better section headers
            lines = doc.page_content.split("\n")
            if len(lines) > 1:
                first_line = lines[0].strip()
                second_line = lines[1].strip()
                section_name = f"{first_line} {second_line}".strip()  # Try combining first two lines for better accuracy
                doc.metadata["section"] = section_name if len(section_name) < 100 else "Unknown Section"

        # ✅ Add text-based chunks to ChromaDB
        chroma_db.add_documents(docs)

        print(f"✅ Indexed {len(docs)} text chunks from {pdf_title}.")
        return docs
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return []

#############################################################################################################
### ✅ UI Layout
##############################################################################################################
app_ui = ui.page_fluid(
    ui.h2("📄 AI-Powered Multi-PDF Analyzer"),
    ui.h6("Upload multiple PDFs and ask questions about them."),
    ui.input_file("file", "Upload PDF Documents", multiple=True, accept=[".pdf"]),  
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
            new_docs.extend(docs)

        if new_docs:
            print(f"✅ Indexed {len(new_docs)} chunks from {len(file_info)} PDFs into ChromaDB.")
        else:
            print("❌ No valid PDFs processed.")

    answer_text = reactive.value("")

    @reactive.effect
    @reactive.event(input.ask)
    def generate_response():
        """Handles GPT-4 response while preserving section structure."""
        query = input.query()
        if not query:
            print("❌ No query provided.")
            answer_text.set("Please enter a valid question.")
            return

        retrieved_docs = retriever.get_relevant_documents(query)
        retrieved_docs.sort(key=lambda doc: doc.metadata.get("chunk_index", 0))

        structured_response = []
        for doc in retrieved_docs:
            title = doc.metadata.get("title", "Unknown Document")
            section = doc.metadata.get("section", "Unknown Section")
            text = doc.page_content.strip()

            structured_response.append(f"📄 **Title: {title}**\n🔹 **Section: {section}**\n📜 {text}")

        formatted_response = "\n\n".join(structured_response)

        result = qa_chain.invoke(query)
        answer_text.set(formatted_response + "\n\n" + "🤖 **AI Summary:**\n" + result["result"])

    @render.text
    def result_text():
        return answer_text.get() if answer_text.get() else "No response yet."

# ✅ Run Shiny App
app = App(app_ui, server)
