import os
import tempfile
import chromadb
import tiktoken
import PyPDF2
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from shiny import App, ui, render, reactive

# Initialize OpenAI API. Set as env variable as it is being hosted.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API Key is missing! Set OPENAI_API_KEY as an environment variable.")

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

#Temp storage for ChromaDB. need to create a tempfile unlike running locally.
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

reset_chromadb()

# Setup LLM. keep the temp lower to create fact based reponse. token=500 is a good balance for cost.
llm = ChatOpenAI(model_name="gpt-4", temperature=0.1, openai_api_key=OPENAI_API_KEY, max_tokens=500)
#########setup functions
# function to Count tokens
def count_tokens(text):
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

# Process PDF and Store in ChromaDB
def process_pdf(file_path):
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        docs = text_splitter.split_documents(pages)

        chroma_db.add_documents(docs)
        print(f"Indexed {len(docs)} text chunks into ChromaDB.")
        return docs
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return []

#############################################################################################################
### UI LAYOUT - Sidebar + Main Content
##############################################################################################################

app_ui = ui.page_fluid(
    ui.layout_sidebar(
        ui.sidebar(
            ui.h3("PDF Analyzer"),
            ui.input_file("file", "Upload PDF", multiple=False, accept=[".pdf"]),
            ui.input_text("query", "Ask a question:", placeholder="Enter your query..."),
            ui.input_action_button("ask", "🔍 Ask AI"),
            ui.output_text("file_info"),
            class_="sidebar"
        ),
       
       # ui.h3("Qurry Result based on the Pdf"),
        ui.panel_well(
            ui.h6("OpenAI LLM, Embeddings, Langchain for retreival and ChromaDB for Vector DB "),
            ui.h6("Since this is a RAG based App., Prompts will result in signficant impact in response.-- "),
            ui.h6(" --Try using keywords from the pdf you uploaded like section headers etc to get accurate response "),
            ui.h6("There is a known issue where a new pdf is uploaded, the ChormaDB might not get cleared. Work is in progress."),
            ui.output_text("ai_summary")
        )
    )
)

# Server 
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

# ✅ Run Shiny App
app = App(app_ui, server)
