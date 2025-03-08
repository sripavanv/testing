import os
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

# ✅ Initialize OpenAI API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API Key is missing! Set OPENAI_API_KEY as an environment variable.")

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# ✅ Temporary storage for ChromaDB
CHROMA_DB_DIR = tempfile.mkdtemp()
chroma_db = None  # Define globally for reset

# ✅ Reset ChromaDB using LangChain
def reset_chromadb():
    global chroma_db
    try:
        if chroma_db is not None:
            chroma_db.delete_collection()
            print("🗑️ ChromaDB cleared.")
    except Exception as e:
        print(f"⚠️ Warning: Could not clear ChromaDB properly: {e}")

    chroma_db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    print(f"✅ ChromaDB reset at {CHROMA_DB_DIR}")

reset_chromadb()

# ✅ Setup LLM with OpenAI
llm = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=OPENAI_API_KEY, max_tokens=500)

# ✅ Count tokens
def count_tokens(text):
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

# ✅ Process PDF and Store in ChromaDB using LangChain
def process_pdf(file_path):
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        docs = text_splitter.split_documents(pages)

        global chroma_db
        chroma_db = Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_DB_DIR)

        print(f"✅ Indexed {len(docs)} text chunks into ChromaDB.")
        return docs
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        return []

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
        
        # ✅ Wrapped AI Response inside a panel container
        ui.panel_well(
            ui.h6("RAG-based LLM using OpenAI, LangChain embeddings, and ChromaDB for vector storage."),
            ui.h6("Since this is a RAG-based implementation, prompts significantly impact responses. Use keywords and section titles from your PDF."),
            ui.h3("📖 AI Summary"),
            ui.output_text("response")
        )
    )
)

# ✅ Server Logic
def server(input, output, session):
    uploaded_file_path = reactive.value("")
    uploaded_file_name = reactive.value("No file uploaded.")
    answer_text = reactive.value("")

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

        global chroma_db
        if chroma_db is None or chroma_db._collection.count() == 0:
            answer_text.set("⚠️ No PDF uploaded. Please upload a file first.")
            return

        print(f"📝 Query received: {query}")

        try:
            # ✅ Use LangChain's retriever instead of manual retrieval
            retriever = chroma_db.as_retriever()
            retrieved_docs = retriever.get_relevant_documents(query)

            if not retrieved_docs:
                answer_text.set("⚠️ No relevant information found.")
                return

            retrieved_text = " ".join([doc.page_content for doc in retrieved_docs])

            total_tokens = count_tokens(query + retrieved_text)
            if total_tokens > 7500:
                answer_text.set("⚠️ Query too long. Try asking a more specific question.")
                return

            # ✅ Use LangChain's RetrievalQA for response generation
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
            result = qa_chain.run(query)

            answer_text.set(result)

        except Exception as e:
            answer_text.set("❌ Error retrieving response.")
            print(f"❌ Error: {e}")

    @render.text
    def response():
        return answer_text.get()

# ✅ Run Shiny App
app = App(app_ui, server)
