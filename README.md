# 📄 RAG Question Answering System

This project is a **Retrieval-Augmented Generation (RAG)** based Question Answering system built using **FastAPI, FAISS, Sentence Transformers, and FLAN-T5**.

The application allows users to upload documents and ask questions based strictly on the document content.

---

## 🚀 Features

- 📂 Upload **PDF or TXT** documents  
- ✂️ Automatic **text chunking**  
- 🧠 Embedding generation using **Sentence Transformers**  
- 🔎 Semantic search with **FAISS vector database**  
- 🤖 Answer generation using a **FLAN-T5 LLM**  
- 📚 Source chunk display for transparency  

---

## 🏗️ System Architecture

The pipeline follows these steps:

1. Document Upload  
2. Text Extraction  
3. Text Cleaning  
4. Chunking  
5. Embedding Generation  
6. Vector Storage (FAISS)  
7. Question Embedding  
8. Similarity Search  
9. LLM Answer Generation  

This ensures answers are grounded in the uploaded document.
## 🟢 Tech Stack

The project uses the following technologies:

- **FastAPI** – Backend API framework  
- **HTML + Tailwind CSS** – Frontend user interface  
- **Sentence Transformers** – For generating text embeddings  
- **FAISS** – Vector database for similarity search  
- **FLAN-T5** – Language model for answer generation  
- **PyPDF2** – For extracting text from PDF documents  


---

## ▶ Run Locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Now open your browser and go to:

👉 **http://127.0.0.1:8000**

---

## 🖥️ How to Use the Application

### 1️⃣ Upload a Document

- Click **Choose File**
- Select a **PDF or TXT** file
- Click **Upload**

The system will:
- Extract text from the document  
- Split it into chunks  
- Generate embeddings  
- Store them in the FAISS vector database  

⏳ Wait until you see:  
**“Processing completed. You can now ask questions.”**

---

### 2️⃣ Ask Questions

- Enter a question related to the uploaded document  

**Example:**
```text
What is ClaSum?
```

- Click **Get Answer**

---

## ⚙️ How Answers Are Generated

When a question is asked:

1. The question is converted into an embedding  
2. FAISS retrieves the most relevant document chunks  
3. These chunks are sent as context to the LLM  
4. The LLM generates an answer **based only on retrieved content**

This reduces hallucinations and keeps answers document-grounded.
