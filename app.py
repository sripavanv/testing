import os
import tempfile
import chromadb
import tiktoken
import PyPDF2
from PIL import Image
import fitz  # PyMuPDF for PDF processing
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from shiny import App, ui, render, reactive

# ✅ Initialize OpenAI API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API Key is missing! Set OPENAI_API_KEY as an environment variable.")

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# ✅ Temporary storage for ChromaDB
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

reset_chromadb()

# ✅ Setup LLM
llm = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=OPENAI_API_KEY, max_tokens=500)

# ✅ Count tokens
def count_tokens(text):
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

# ✅ Process PDF and Store Page Numbers
def process_pdf(file_path):
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        docs = text_splitter.split_documents(pages)

        for i, doc in enumerate(docs):
            doc.metadata["page"] = pages[i // len(pages)].metadata["page"]

        chroma_db.add_documents(docs)
        print(f"✅ Indexed {len(docs)} text chunks into ChromaDB.")
        return docs
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        return []

# ✅ Extract Section Image from PDF
def extract_section_image(file_path, page_number):
    try:
        doc = fitz.open(file_path)
        page = doc[page_number - 1]
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return img
    except Exception as e:
        print(f"❌ Error extracting image: {e}")
        return None

#############################################################################################################
### 🚀 UI LAYOUT - Sidebar + Main Content
##############################################################################################################

app_ui = ui.page_fluid(
    ui.layout_sidebar(
        ui.sidebar(
            ui.h3("📄 PDF Analyzer"),
            ui.input_file("file", "Upload PDF", multiple=False, accept=[".pdf"]),
            ui.input_text("query", "Ask a question:", placeholder="Enter your query..."),
            ui.input_action_button("ask", "🔍 Ask AI"),
            ui.output_text("file_info"),
            class_="sidebar"
        ),
        # ✅ The main content should be placed directly in layout_sidebar()
        ui.h3("📖 AI Response"),
        ui.output_text("response"),
        ui.h3("📸 Section Reference"),
        ui.output_image("section_image")
    )
)

# ✅ Server Logic
def server(input, output, session):
    uploaded_file_path = reactive.value("")
    uploaded_file_name = reactive.value("No file uploaded.")
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
            uploaded_file_name.set(f"✅ Uploaded: {file_info[0]['name']}")
            print("✅ PDF uploaded and processed successfully.")
        else:
            uploaded_file_name.set("❌ Failed to process PDF.")

    @render.text
    def file_info():
        return uploaded_file_name.get()

    @reactive.effect
    @reactive.event(input.ask)
    def generate_response():
        query = input.query()
        if not query:
            answer_text.set("⚠️ Please enter a valid question.")
            return

        if chroma_db._collection.count() == 0:
            answer_text.set("⚠️ No PDF uploaded. Please upload a file first.")
            return

        print(f"📝 Query received: {query}")

        try:
            retrieved_docs = chroma_db.as_retriever().get_relevant_documents(query)
            if not retrieved_docs:
                answer_text.set("⚠️ No relevant information found.")
                return

            retrieved_text = " ".join([doc.page_content for doc in retrieved_docs])
            retrieved_page.set(retrieved_docs[0].metadata["page"])

            total_tokens = count_tokens(query + retrieved_text)
            if total_tokens > 7500:
                answer_text.set("⚠️ Query too long. Try asking a more specific question.")
                return

            result = RetrievalQA.from_chain_type(llm=llm, retriever=chroma_db.as_retriever()).invoke(query)
            answer_text.set(result["result"] if "result" in result else str(result))

        except Exception as e:
            answer_text.set("❌ Error retrieving response.")
            print(f"❌ Error: {e}")

    @render.text
    def response():
        return answer_text.get()

    @render.image
    def section_image():
        file_path = uploaded_file_path.get()
        page_number = retrieved_page.get()  # ✅ Ensure `.get()` is used for reactive values

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
