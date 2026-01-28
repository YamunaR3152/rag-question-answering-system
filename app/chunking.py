import re

def smart_overlap(text, overlap):
    """Ensure overlap starts at a word boundary"""
    if len(text) <= overlap:
        return text

    overlap_text = text[-overlap:]

    # Cut to nearest space to avoid broken words
    first_space = overlap_text.find(" ")
    if first_space != -1:
        overlap_text = overlap_text[first_space+1:]

    return overlap_text.strip()


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150):
    if not text or len(text) < 200:
        return []

    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Split into sentences more reliably
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()

        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    # Add SAFE overlap (no word cutting)
    final_chunks = []
    for i in range(len(chunks)):
        chunk = chunks[i]

        if i > 0:
            overlap_text = smart_overlap(chunks[i-1], overlap)
            chunk = overlap_text + " " + chunk

        final_chunks.append(chunk.strip())

    return final_chunks
