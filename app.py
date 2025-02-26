from shiny import App, ui, render, reactive
import pandas as pd
import openai
import pdfplumber
import base64
import io 
import os
from PIL import Image, UnidentifiedImageError
import shinyswatch  # For themes

# ✅ Ensure OpenAI API Key is set correctly
if os.getenv("API_VAR"):
    os.environ["OPENAI_API_KEY"] = os.getenv("API_VAR")

# ✅ Global storage for extracted content
extracted_data = reactive.value({"text": "", "tables": [], "images": []})
answer_text = reactive.value("")  # Stores the AI response
relevant_tables = reactive.value([])  # Stores only relevant tables
relevant_images = reactive.value([])  # Stores only relevant images

def extract_text_tables_images_from_pdfs(files):
    """Extracts text, tables, and images from uploaded PDFs."""
    text, tables, images = "", [], []

    for _ in files:
        with pdfplumber.open(_) as pdf:
            for page in pdf.pages:
                if page.extract_text():
                    text += page.extract_text() + "\n"

                # ✅ Extract tables
                page_tables = page.extract_tables()
                for table in page_tables:
                    if table and len(table) > 1:
                        df = pd.DataFrame(table).drop(0).reset_index(drop=True)
                        tables.append(df)

                # ✅ Extract images
                for img in page.images:
                    try:
                        img_data = img["stream"].get_data()
                        image = Image.open(io.BytesIO(img_data))
                        images.append(image)
                    except UnidentifiedImageError:
                        print("Skipping invalid image in PDF")

    extracted_data.set({"text": text, "tables": tables, "images": images})
    print("✅ PDF processing complete: Text extracted & stored.")

def answer_question(query):
    """Retrieves relevant chunks and generates an answer with GPT-4."""
    data = extracted_data.get()
    context = data["text"][:5000]  # Limit context for API call

    if not context:
        return "No relevant text found in PDF."

    # ✅ Use OpenAI API
    client = openai.OpenAI(api_key=os.getenv("API_VAR"))

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Answer based on:\n\n{context}\n\nQ: {query}"}
        ]
    )

    answer = response.choices[0].message.content
    print(f"🔍 OpenAI Response: {answer}")

    # ✅ Find relevant tables
    matching_tables = [table for table in data["tables"] if any(query.lower() in str(cell).lower() for cell in table.to_numpy().flatten())]
    relevant_tables.set(matching_tables)

    # ✅ Assume all images are relevant for now
    relevant_images.set(data["images"])

    return answer

# ✅ Define UI
app_ui = ui.page_fluid(
    ui.TagList(  
        ui.h1("Python Shiny PDF AI Assistant"),
        
        ui.layout_sidebar(
            ui.sidebar(  
                ui.input_file("pdf_file", "Upload PDF(s)", multiple=True, accept=[".pdf"]),
                ui.input_text("query", "Enter your question"),
                ui.input_action_button("ask", "Ask")  # ✅ Now ensures query runs only when clicked
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
        """Extract data when PDFs are uploaded"""
        files = input.pdf_file()
        if files:
            print("📂 PDF Uploaded: Processing...")
            extract_text_tables_images_from_pdfs([f["datapath"] for f in files])

    @reactive.effect
    @reactive.event(input.ask)  # ✅ Now waits for the button click!
    def update_answer():
        """Answer questions only when the 'Ask' button is clicked"""
        query = input.query()
        if query:
            print(f"📝 Processing Query: {query}")  # ✅ Debugging print
            answer_text.set(answer_question(query))  # ✅ Store AI response in reactive value

    # ✅ Define response output
    @render.text
    def response():
        return answer_text.get() if answer_text.get() else "No response yet."

    output.response = response

    # ✅ Define table output
    @render.table
    def table_output():
        """Display only relevant tables after asking a question."""
        tables = relevant_tables.get()
        return tables[0] if tables else pd.DataFrame({"Message": ["No relevant tables found."]})

    output.table_output = table_output

    # ✅ Define image output
    @render.ui
    def image_output():
        """Display only relevant images after asking a question."""
        images = relevant_images.get()
        image_tags = [f'<img src="data:image/png;base64,{base64.b64encode(io.BytesIO().getvalue()).decode()}" width="200px" style="margin:5px;">' for img in images]

        return ui.HTML("".join(image_tags) if image_tags else "No relevant images found.")

    output.image_output = image_output

# ✅ Run the app
app = App(app_ui, server)
