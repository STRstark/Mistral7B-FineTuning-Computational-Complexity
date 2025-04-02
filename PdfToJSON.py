import fitz  
import re
import uuid
import json
import yake
import os 

def extract_keywords(text, max_keywords=5):
    kw_extractor = yake.KeywordExtractor(n=1, top=max_keywords)
    keywords = kw_extractor.extract_keywords(text)
    return [kw[0] for kw in keywords]

def clean(text):
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def chunkDeviding(text, chunk_size=500):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def GiveDataStructure(chunks, metadata):
    
    structured_chunks = []

    for chunk in chunks:
        structured_chunks.append({
            "id": str(uuid.uuid4()),
            "title": metadata.get("title", ""),
            "chunk_text": chunk,
            "metadata": {
                "author": metadata.get("author", ""),
                "keywords": extract_keywords(chunk , 10)
            }
        })

    return structured_chunks

def StoreDataFromPdfToJSON(pdf_path , destination):

    doc = fitz.open(pdf_path)
    full_text = ""

    for page in doc:
        full_text += page.get_text("text") + " "

    cleanText = clean(full_text)    

    chunks = chunkDeviding(cleanText)
    
    structure = GiveDataStructure(chunks,doc.metadata)
    
    output_path = f"{destination}/{str(uuid.uuid4())}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(structure, f, ensure_ascii=False, indent=4)

def PdfToJSON(pdf_fiels ,json_path ):
    pdfs = [f for f in os.listdir(pdf_fiels) if f.endswith(".pdf")]
    for pdf in pdfs :
        StoreDataFromPdfToJSON(pdf_fiels+f"/{pdf}" , json_path)

PdfToJSON("pdfFiles" , "JSONFIles")
# pdf_fiels = "pdfFiles"
# json_path = "JSONFIles"

