import chromadb
from mcp.server.fastmcp import FastMCP

DB_FOLDER = "/Users/johanneswiebe/dev/docstore/chroma_store"

# Connect to persistent Chroma DB
client = chromadb.PersistentClient(path=DB_FOLDER)
collection = client.get_or_create_collection("pdf_docs")

mcp = FastMCP("docstore-mcp")

@mcp.tool()
def search(query: str, n_results: int = 5) -> dict:
    """
    Search indexed PDF documents for relevant chunks.
    Returns a list of document chunks with source filenames.
    """
    if not query:
        return {"error": "query cannot be empty"}
    results = collection.query(query_texts=[query], n_results=n_results)
    matches = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        matches.append({
            "source": meta.get("source", "unknown"),
            "text": doc
        })
    return {"results": matches}

if __name__ == '__main__':
    mcp.run()
