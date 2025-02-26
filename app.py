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

# ✅ Ensure OpenAI API Key is set correctly
if os.getenv("API_VAR"):
    os.environ["OPENAI_API_KEY"] = os.getenv("API_VAR")

# ✅ Initialize ChromaDB (Persistent storage)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# ✅ Define embedding function using OpenAI
openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ Create or load collection
collection = chroma_client.get_or_create_collection(name="pdf_embeddings", embedding_function=openai_ef)

# ✅ Holds extracted data
extracted_data = reactive.value({"text": [], "tables": [], "images": []})
answer_text = reactive.value("")  # Stores GPT response
relevant_tables = reactive.value([])  # Stores relevant tables
relevant_images = reactive.value([])  # Stores relevant images

def chunk_text(text, chunk_size=500, overlap=100):
    """Splits text into overlapping chunks for better storage & retrieval."""
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]
    return chunks

def extract_text_tables_images_from_pdfs(files):
    """Efficiently extracts text, tables, and images, storing embeddings."""
    text_chunks, tables, images = [], [], []

    for file in files:
        with pdfplumber.open(file) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    chunks = chunk_text(text)
                    text_chunks.extend(chunks)

                    # ✅ Store text chunks as embeddings
                    for i, chunk in enumerate(chunks):
                        collection.add(
                            ids=[f"{file}-p{page_num}-chunk{i}"],
                            documents=[chunk],
                            metadatas=[{"type": "text"}]
                        )

                # ✅ Extract tables
                for i, table in enumerate(page.extract_tables()):
                    if table and len(table) > 1:
                        df = pd.DataFrame(table).drop(0).reset_index(drop=True)
                        tables.append(df)
                        table_text = "\n".join([" | ".join(map(str, row)) for row in df.values])
                        collection.add(
                            ids=[f"{file}-p{page_num}-table{i}"],
                            documents=[table_text],
                            metadatas=[{"type": "table"}]
                        )

                # ✅ Extract & batch embed images
                batch_images, image_ids = [], []
                for i, img in enumerate(page.images):
                    try:
                        img_data = img["stream"].get_data()
                        image = Image.open(io.BytesIO(img_data))
                        images.append(image)

                        # ✅ Convert image to base64 for batch embedding
                        buffered = io.BytesIO()
                        image.save(buffered, format="PNG")
                        img_base64 = base64.b64encode(buffered.getvalue()).decode()

                        batch_images.append(img_base64)
                        image_ids.append(f"{file}-p{page_num}-image{i}")

                    except UnidentifiedImageError:
                        continue

                # ✅ Send batch image embedding request
                if batch_images:
                    response = openai.embeddings.create(model="image-embedding-clip", input=batch_images)
                    image_embeddings = [res["embedding"] for res in response["data"]]
                    collection.add(
                        ids=image_ids,
                        embeddings=image_embeddings,
                        metadatas=[{"type": "image"}] * len(image_ids)
                    )

    extracted_data.set({"text": text_chunks, "tables": tables, "images": images})
    print("✅ PDF processing complete (Optimized).")

def answer_question(query):
    """Retrieves relevant text, tables, and images from ChromaDB and generates an answer with GPT-4."""
    
    # ✅ Retrieve top 5 relevant text chunks
    results = collection.query(
        query_texts=[query],
        n_results=5,
        where={"type": "text"}  # ✅ Only search text embeddings
    )
    relevant_chunks = results["documents"][0] if results["documents"] else []

    if not relevant_chunks:
        return "No relevant text found in PDF."

    context = "\n\n".join(relevant_chunks)[:5000]  # Limit to 5000 chars for GPT-4

    # ✅ Retrieve relevant tables
    matching_tables = []
    for table in extracted_data.get()["tables"]:
        if any(query.lower() in str(cell).lower() for cell in table.to_numpy().flatten()):
            matching_tables.append(table)

    relevant_tables.set(matching_tables)

    # ✅ Retrieve relevant images
    image_results = collection.query(
        query_texts=[query],
        n_results=3,
        where={"type": "image"}  # ✅ Only search image embeddings
    )
    relevant_image_ids = image_results["ids"][0] if image_results["ids"] else []

    matching_images = []
    for image_id in relevant_image_ids:
        page_num = int(image_id.split("-p")[1].split("-image")[0])
        images_on_page = extracted_data.get()["images"]
        if page_num < len(images_on_page):
            matching_images.append(images_on_page[page_num])

    relevant_images.set(matching_images)

    # ✅ Call OpenAI for answer
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
                ui.input_action_button("ask", "Ask")
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

    @reactive.effect
    @reactive.event(input.ask)
    def update_answer():
        """Generate an answer ONLY when the button is clicked."""
        query = input.query()
        if query:
            print(f"📝 Query: {query}")
            answer_text.set(answer_question(query))

    @render.text
    def response():
        return answer_text.get() if answer_text.get() else "No response yet."

    output.response = response  

    @render.table
    def table_output():
        tables = relevant_tables.get()
        return tables[0] if tables else pd.DataFrame({"Message": ["No relevant tables found."]})

    output.table_output = table_output  

    @render.ui
    def image_output():
        images = relevant_images.get()
        image_tags = [f'<img src="data:image/png;base64,{base64.b64encode(img.tobytes()).decode()}" width="200px" style="margin:5px;">' for img in images]
        return ui.HTML("".join(image_tags) if image_tags else "No relevant images found.")

    output.image_output = image_output  

# ✅ Run the app
app = App(app_ui, server)
