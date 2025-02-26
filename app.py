from shiny import App, ui, render, reactive
import pandas as pd
import openai
import pdfplumber
import base64
import io 
import os
import chromadb
from chromadb.utils import embedding_functions
from PIL import Image, UnidentifiedImageError
import shinyswatch  # For themes

# ✅ Initialize ChromaDB (Persistent storage)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# ✅ Define embedding function using OpenAI
openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=os.getenv("API_VAR"))

# ✅ Create or load collection
collection = chroma_client.get_or_create_collection(name="pdf_embeddings", embedding_function=openai_ef)

# ✅ Holds extracted data
extracted_data = reactive.value({"text": [], "tables": [], "images": []})
answer_text = reactive.value("")  # Stores GPT response
relevant_tables = reactive.value([])  # Stores relevant tables
relevant_images = reactive.value([])  # Stores relevant images

def extract_text_tables_images_from_pdfs(files):
    """Extracts text, tables, and images from PDFs and stores embeddings in ChromaDB."""
    text_chunks, tables, images = [], [], []

    for file in files:
        with pdfplumber.open(file) as pdf:
            for page_num, page in enumerate(pdf.pages):
                if page.extract_text():
                    chunk_text = page.extract_text()
                    text_chunks.append(chunk_text)

                    # ✅ Chunking text into 500-word overlapping windows
                    words = chunk_text.split()
                    chunk_size = 100
                    overlap = 20
                    for i in range(0, len(words), chunk_size - overlap):
                        chunk = " ".join(words[i:i + chunk_size])
                        chunk_id = f"{file}-p{page_num}-chunk{i}"
                        collection.add(ids=[chunk_id], documents=[chunk])

                # ✅ Extract tables
                page_tables = page.extract_tables()
                for i, table in enumerate(page_tables):
                    if table and len(table) > 1:
                        df = pd.DataFrame(table).drop(0).reset_index(drop=True)
                        tables.append(df)
                        
                        # ✅ Store tables as text embeddings
                        table_text = "\n".join([" | ".join(map(str, row)) for row in df.values])
                        table_id = f"{file}-p{page_num}-table{i}"
                        collection.add(ids=[table_id], documents=[table_text])

                # ✅ Extract images
                for i, img in enumerate(page.images):
                    try:
                        img_data = img["stream"].get_data()
                        image = Image.open(io.BytesIO(img_data))
                        images.append(image)
                    except UnidentifiedImageError:
                        print("Skipping invalid image in PDF")

    extracted_data.set({"text": text_chunks, "tables": tables, "images": images})
    print("✅ PDF processing complete.")

def answer_question(query):
    """Retrieves relevant chunks from ChromaDB and generates an answer with GPT-4."""
    
    # ✅ Retrieve top 5 relevant chunks from ChromaDB
    results = collection.query(query_texts=[query], n_results=5)
    relevant_chunks = results["documents"][0] if results["documents"] else []

    if not relevant_chunks:
        return "No relevant text found in PDF."

    context = "\n\n".join(relevant_chunks)[:5000]  # Limit to 5000 chars for GPT-4

    # ✅ Find relevant tables
    matching_tables = []
    for table in extracted_data.get()["tables"]:
        if any(query.lower() in str(cell).lower() for cell in table.to_numpy().flatten()):
            matching_tables.append(table)

    relevant_tables.set(matching_tables)  # Store relevant tables

    # ✅ Assume all images are relevant for now
    relevant_images.set(extracted_data.get()["images"])

    # ✅ Call OpenAI for answer
    client = openai.OpenAI(api_key=os.getenv("API_VAR"))

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Answer based on:\n\n{context}\n\nQ: {query}"}
        ]
    )

    answer = response.choices[0].message.content
    print(f"🔍 GPT-4 Response: {answer}")  
    return answer

# ✅ Define UI
app_ui = ui.page_fluid(
    ui.TagList(
        ui.h1("Python Shiny PDF AI Assistant"),
        
        ui.layout_sidebar(
            ui.sidebar(  
                ui.input_file("pdf_file", "Upload PDF(s)", multiple=True, accept=[".pdf"]),
                ui.input_text("query", "Enter your question"),
                ui.input_action_button("ask", "Ask")  # ✅ Now only runs on button click!
            ),

            ui.card(  
                ui.h3("Response"),
                ui.output_text("response"),
                ui.h3("Relevant Tables"),
                ui.output_table("table_output"),
                ui.h3("Relevant Images"),
                ui.output_ui("image_output")
            )
        )
    ),
    theme=shinyswatch.theme.darkly()
)

# ✅ Define Server Logic
def server(input, output, session):
    """Server logic for handling user interactions"""

    @reactive.effect
    def process_files():
        """Process uploaded PDFs and store embeddings."""
        files = input.pdf_file()
        if files:
            print("📂 Processing PDFs...")
            extract_text_tables_images_from_pdfs([f["datapath"] for f in files])

    @reactive.event(input.ask)  # ✅ Now waits for the button click!
    def update_answer():
        """Generate an answer ONLY when the button is clicked."""
        query = input.query()
        if query:
            print(f"📝 Query: {query}")
            answer_text.set(answer_question(query))

    # ✅ Define response output
    @render.text
    def response():
        return answer_text.get() if answer_text.get() else "No response yet."

    output.response = response  

    # ✅ Define table output
    @render.table
    def table_output():
        """Display relevant tables."""
        tables = relevant_tables.get()
        return tables[0] if tables else pd.DataFrame({"Message": ["No relevant tables found."]})

    output.table_output = table_output  

    # ✅ Define image output
    @render.ui
    def image_output():
        """Display relevant images."""
        images = relevant_images.get()
        image_tags = []

        for img in images:
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            image_tags.append(f'<img src="data:image/png;base64,{img_base64}" width="200px" style="margin:5px;">')

        return ui.HTML("".join(image_tags) if image_tags else "No relevant images found.")

    output.image_output = image_output  

# ✅ Run the app
app = App(app_ui, server)
