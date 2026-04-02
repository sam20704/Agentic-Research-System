import os
import fitz  # PyMuPDF

def load_pdfs(folder_path):
    documents = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)

            doc = fitz.open(file_path)
            text = ""

            for page in doc:
                text += page.get_text()

            documents.append({
                "source": filename,
                "content": text
            })

    return documents