from PyPDF2 import PdfReader
import re
from app.chunking import chunk_text
from app.embeddings import generate_embeddings


def clean_text(text: str) -> str:
    if not text:
        return ""

    # Fix camelCase joins
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)

    # Fix missing spaces after punctuation
    text = re.sub(r"(?<=[.,;:])(?=[A-Za-z])", " ", text)

    # Remove broken line merges from PDFs
    text = text.replace("\n", " ")

    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()



def extract_text_from_file(file_path: str) -> str:
    if file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += " " + page_text

        return clean_text(text)

    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return clean_text(f.read())

    else:
        raise ValueError("Unsupported file format")


def ingest_document(file_path: str, vector_store):
    print("📄 Extracting text...")
    text = extract_text_from_file(file_path)

    print("✂️ Chunking text...")
    chunks = chunk_text(text)

    print(f"🔢 Created {len(chunks)} chunks")

    print("🧠 Generating embeddings...")
    embeddings = generate_embeddings(chunks)

    print("💾 Storing in vector DB...")
    vector_store.add(embeddings, chunks)

    print("✅ Ingestion finished")
    return len(chunks)
