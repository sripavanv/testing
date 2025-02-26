from shiny import App, ui, render, reactive
import pandas as pd
import openai
import pdfplumber
import base64
import io 
import os
from PIL import Image, UnidentifiedImageError
import shinyswatch  # For themes

# OpenAI API Key (Replace with your actual key)
#OPENAI_API_KEY = "sk-pr"
os.environ["OPENAI_API_KEY"] = os.getenv("API_VAR")
# Global storage for extracted content

## holds reactive value that is dictionary. dictionary is initialized while also initializing a reactive.value.
extracted_data = reactive.value({"text": "", "tables": [], "images": []})
###reactive value holding a string
answer_text = reactive.value("")  # Stores the AI response
###reactive value holdin a list
relevant_tables = reactive.value([])  # Stores only relevant tables
relevant_images = reactive.value([])  # Stores only relevant images


def extract_text_tables_images_from_pdfs(files):
    """Extracts text, tables, and images from uploaded PDFs."""
    ## we are initializing again so in each loop, values dont get mixed up and they can be kept in thier own page
    text, tables, images = "", [], []

    for _ in files:
        with pdfplumber.open(_) as pdf:
            for page in pdf.pages:
                if page.extract_text():
                    text += page.extract_text() + "\n"

                # pdf plumber Extract tables
                page_tables = page.extract_tables()
                for table in page_tables:
                    if table and len(table) > 1:
                        df = pd.DataFrame(table)
                        #df.columns = df.iloc[0]  # Use first row as column names
                        #df = df[1:].reset_index(drop=True)
                        df = df.drop(0).reset_index(drop=True)  ##drop row 1 which is now headers
                        tables.append(df)

                # pdf plumber Extract images
                for img in page.images:
                    try:
                        img_data = img["stream"].get_data()  ##img holds the dictionary containing metadata about an image inside a PDF. 
                        #img[stream] holds the image data
                        #use pil package to covert bytes into file
                        image = Image.open(io.BytesIO(img_data))
                        images.append(image)
                    except UnidentifiedImageError:
                        print("Skipping invalid image in PDF")

    extracted_data.set({"text": text, "tables": tables, "images": images})
    print("✅ PDF processing complete: Text extracted & stored.")


def answer_question(query):
    """Retrieves relevant chunks and generates an answer with GPT-4."""
    data = extracted_data.get() #### retrive the current values using get()
    context = data["text"][:5000]  # Limit context for API call

    if not context:
        return "No relevant text found in PDF.", [], []

    # use OpenAI API
    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Answer based on:\n\n{context}\n\nQ: {query}"}
        ]
    )

    answer = response.choices[0].message.content
    print(f" OpenAI Response: {answer}")  # ✅ Debugging print

    # Find relevant tables. it is matched using keyworkds from querry
    matching_tables = []
    for table in data["tables"]:
        if any(query.lower() in str(cell).lower() for cell in table.to_numpy().flatten()):
            matching_tables.append(table)

    # store only relevant tables
    relevant_tables.set(matching_tables)

    # Assume all images are relevant (since images don’t contain searchable text)
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
                ui.input_action_button("ask", "Ask")
            ),

            ui.card(  
                ui.h3("Response"),
                ui.output_text("response"),  # ✅ Fixed Output
                ui.h3("Relevant Tables"),
                ui.output_table("table_output"),  # ✅ Fixed Output
                ui.h3("Relevant Images"),
                ui.output_ui("image_output")  # ✅ Fixed Output
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
    def update_answer():
        """Answer questions when button is clicked"""
        if input.ask():
            query = input.query()
            if query and extracted_data.get()["text"]:
                print(f"📝 Processing Query: {query}")  # ✅ Debugging print
                answer = answer_question(query)
                answer_text.set(answer)  # ✅ Store AI response in reactive value

    # ✅ Properly define response output
    @render.text
    def response():
        return answer_text.get() if answer_text.get() else "No response yet."

    output.response = response  # ✅ Assign output

    # ✅ Properly define table output
    @render.table
    def table_output():
        """Display only relevant tables after asking a question."""
        tables = relevant_tables.get()
        return tables[0] if tables else pd.DataFrame({"Message": ["No relevant tables found."]})

    output.table_output = table_output  # ✅ Assign output

    # ✅ Properly define image output
    @render.ui
    def image_output():
        """Display only relevant images after asking a question."""
        images = relevant_images.get()
        image_tags = []

        for img in images:
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            image_tags.append(f'<img src="data:image/png;base64,{img_base64}" width="200px" style="margin:5px;">')

        return ui.HTML("".join(image_tags) if image_tags else "No relevant images found.")

    output.image_output = image_output  # ✅ Assign output


# ✅ Run the app
app = App(app_ui, server)
#app.run()
