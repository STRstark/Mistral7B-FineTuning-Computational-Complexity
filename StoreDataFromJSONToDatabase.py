import chromadb
import json
from sentence_transformers import SentenceTransformer
import os
from tqdm import tqdm

print("Loading SentenceTransformer model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Model loaded successfully.")

print("Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(name="research_chunks")
print("Connected to ChromaDB.")

DATA_PATH = "/home/moahamdreza/Desktop/Ai/JSONFIles"
json_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".json")]
print(f"{len(json_files)} JSON files found in {DATA_PATH}.")

for file_name in tqdm(json_files, desc="Processing JSON files"):
    file_path = os.path.join(DATA_PATH, file_name)
    print(f"\nProcessing file: {file_name}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            documents = json.load(f)
    except Exception as e:
        print(f"Error reading {file_name}: {e}")
        continue

    if not isinstance(documents, list):
        print(f"Skipped {file_name}: JSON root is not a list.")
        continue

    for idx, doc in enumerate(documents):
        if not all(key in doc for key in ["chunk_text", "id", "title", "metadata"]):
            print(f"Skipped chunk {idx} in {file_name}: missing required keys.")
            continue

        try:
            text = doc["chunk_text"]
            chunk_id = doc["id"]
            title = doc["title"]
            metadata = doc.get("metadata", {})

            author = metadata.get("author", "Unknown")
            keywords = metadata.get("keywords", [])
            keywords_str = ", ".join(keywords)

            print(f"Embedding chunk {idx + 1}/{len(documents)} in {file_name}...")
            vector = model.encode(text).tolist()

            collection.add(
                ids=[chunk_id],
                embeddings=[vector],
                metadatas=[{
                    "article_id": chunk_id,
                    "title": title,
                    "author": author,
                    "keywords": keywords_str
                }]
            )
            print(f"Chunk {chunk_id} inserted successfully.")

        except Exception as e:
            print(f"Error inserting chunk {chunk_id}: {e}")
