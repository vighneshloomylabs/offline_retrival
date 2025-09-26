from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Read and split context file
with open("info.txt", "r", encoding="utf-8") as f:
    docs = f.read().split("\n\n")  # split into paragraphs

# Create embeddings
embeddings = model.encode(docs, convert_to_numpy=True)

# Build FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

def semantic_search(query: str, k: int = 3):
    # Ensure k is within bounds [3, 7]
    k = max(3, min(7, k))

    # Encode query
    query_embedding = model.encode([query], convert_to_numpy=True)

    # Search
    distances, indices = index.search(query_embedding, k)

    # Return matching docs
    results = [docs[i] for i in indices[0]]
    return results

# ---- Example interactive usage ----
if __name__ == "__main__":
    query = input("Enter your query: ")
    
    # Ask for k, but make it optional
    k_input = input("Enter number of results (3–7, default=3): ")
    k = int(k_input) if k_input.isdigit() else 3

    results = semantic_search(query, k)
    
    print("\nTop Results:")
    for idx, res in enumerate(results, 1):
        print(f"\n[{idx}] {res}")