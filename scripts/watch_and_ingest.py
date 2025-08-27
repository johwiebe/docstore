#!/usr/bin/env python3
import os
import time
import hashlib
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import TokenTextSplitter
from pypdf import PdfReader
from pathlib import Path

# --- CONFIG ---
WATCH_FOLDER = Path("/Users/johanneswiebe/cloud/books")
DB_FOLDER = "/Users/johanneswiebe/dev/docstore/chroma_store"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
BATCH_SIZE = 10

# Chroma client
client = chromadb.PersistentClient(path=DB_FOLDER)
collection = client.get_or_create_collection("pdf_docs")

# splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splitter = TokenTextSplitter(
    chunk_size=1000,  # max 1000 tokens per chunk
    chunk_overlap=50,
    encoding_name="cl100k_base"  # matches OpenAI embeddings
)

# Get already ingested files
existing_sources = set()
res = collection.get(include=["metadatas"])
for meta in res["metadatas"]:
    if "source" in meta:
        existing_sources.add(meta["source"])

print("Already ingested:", existing_sources)

def hash_file(path):
    """Simple hash to avoid duplicate insertions."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

def ingest_pdf(path):
    if path.name in existing_sources:
        print(f"Skipping already ingested PDF: {path}")
        return
    print(f"[INFO] Ingesting {path}")
    reader = PdfReader(path)
    raw_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text + "\n"

    chunks = splitter.split_text(raw_text)
    doc_id = hash_file(path)
    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": os.path.basename(path)} for _ in chunks]

    for i in range(0, len(chunks), BATCH_SIZE):
        batch_docs = chunks[i:i+BATCH_SIZE]
        batch_ids = ids[i:i+BATCH_SIZE]
        batch_metas = metadatas[i:i+BATCH_SIZE]

        collection.add(
            documents=batch_docs,
            ids=batch_ids,
            metadatas=batch_metas
        )
        print(f"[INFO] Added batch {i//BATCH_SIZE+1} ({len(batch_docs)} chunks)")

class PDFHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.lower().endswith(".pdf"):
            time.sleep(1)  # wait for file write to finish
            ingest_pdf(event.src_path)

if __name__ == "__main__":
    os.makedirs(WATCH_FOLDER, exist_ok=True)

    for pdf_file in WATCH_FOLDER.glob("*pdf"):
        ingest_pdf(pdf_file)

    observer = Observer()
    event_handler = PDFHandler()
    observer.schedule(event_handler, str(WATCH_FOLDER), recursive=False)
    observer.start()
    print(f"[INFO] Watching {WATCH_FOLDER} for PDFs...")
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

