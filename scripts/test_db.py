import chromadb

DB_FOLDER = "./chroma_store"  # adjust if using absolute path

# Connect to the persistent Chroma client
client = chromadb.PersistentClient(path=DB_FOLDER)

# List all collections
collections = client.list_collections()
print("Collections:", [c.name for c in collections])

# Get your PDF collection
collection = client.get_collection("pdf_docs")

res = collection.get(include=["documents", "metadatas"], limit=10)
for doc, meta in zip(res["documents"], res["metadatas"]):
    print("Source:", meta.get("source", "unknown"))
    print("Text snippet:", doc[:200], "...\n")

