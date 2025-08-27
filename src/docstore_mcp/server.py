import chromadb
from mcp.server.fastmcp import FastMCP
from typing import Optional

DB_FOLDER = "/Users/johanneswiebe/dev/docstore/chroma_store"

# Connect to persistent Chroma DB
client = chromadb.PersistentClient(path=DB_FOLDER)
collection = client.get_or_create_collection("pdf_docs")

mcp = FastMCP("docstore-mcp")

@mcp.tool()
def list_documents() -> dict:
    """
    Get a list of all indexed documents.
    Returns a list of document filenames that have been ingested.
    """
    try:
        # Get all documents to extract unique sources
        results = collection.get(include=["metadatas"])
        sources = set()
        for meta in results["metadatas"]:
            if "source" in meta:
                sources.add(meta["source"])
        
        return {
            "documents": sorted(list(sources)),
            "total_documents": len(sources)
        }
    except Exception as e:
        return {"error": f"Failed to retrieve documents: {str(e)}"}

@mcp.tool()
def search(query: str, n_results: int = 5, document: Optional[str] = None) -> dict:
    """
    Search indexed PDF documents for relevant chunks.
    
    Args:
        query: The search query text
        n_results: Maximum number of results to return (default: 5)
        document: Optional specific document filename to search within
    """
    if not query:
        return {"error": "query cannot be empty"}
    
    try:
        # If a specific document is specified, filter by metadata
        if document:
            # First verify the document exists
            all_results = collection.get(include=["metadatas"])
            existing_sources = set()
            for meta in all_results["metadatas"]:
                if "source" in meta:
                    existing_sources.add(meta["source"])
            
            if document not in existing_sources:
                return {"error": f"Document '{document}' not found. Available documents: {sorted(list(existing_sources))}"}
            
            # Search with metadata filter
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                where={"source": document}
            )
        else:
            # Search across all documents
            results = collection.query(
                query_texts=[query],
                n_results=n_results
            )
        
        matches = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            matches.append({
                "source": meta.get("source", "unknown"),
                "text": doc
            })
        
        return {
            "results": matches,
            "query": query,
            "document_filter": document,
            "total_results": len(matches)
        }
    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}

@mcp.tool()
def get_document_info(document: str) -> dict:
    """
    Get information about a specific document including chunk count and metadata.
    
    Args:
        document: The filename of the document to get info for
    """
    try:
        # Get all chunks for the specific document
        results = collection.get(
            where={"source": document},
            include=["metadatas", "embeddings"]
        )
        
        if not results["ids"]:
            return {"error": f"Document '{document}' not found"}
        
        chunk_count = len(results["ids"])
        
        return {
            "document": document,
            "chunk_count": chunk_count,
            "total_chunks": chunk_count,
            "has_embeddings": len(results["embeddings"]) > 0 if results["embeddings"] else False
        }
    except Exception as e:
        return {"error": f"Failed to get document info: {str(e)}"}

if __name__ == '__main__':
    mcp.run()
