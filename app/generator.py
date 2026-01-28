from transformers import pipeline

qa_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=256
)

def generate_answer(question, contexts):
    context_text = " ".join(contexts)

    prompt = f"""

You are an academic question answering system.

Answer the question using ONLY the provided research paper context.
Give a clear, complete explanation in 2-4 sentences.
Rules:
- Do NOT guess
- Do NOT add outside knowledge
- If the answer is unclear, say: "Answer not found in document"
Context:
{context_text}

Question:
{question}

Detailed Answer:
"""

    result = qa_pipeline(prompt)[0]["generated_text"]
    return result.strip()
