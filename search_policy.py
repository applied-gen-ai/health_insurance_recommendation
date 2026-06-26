import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from uuid import uuid4
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
import pickle
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
import os


embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


dimension = 384  # Default dimension for all-MiniLM-L6-v2
index = faiss.IndexFlatL2(dimension)

index_path = "./faiss_index"  # Path where FAISS index is stored




def search_with_metadata_key_value(query, provider, policy, k=10):
    if os.path.exists(index_path):
        print("FAISS index found! Loading...")
        vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        print("FAISS index not found. Please create and store the index first.")
        return []

    # Step 1: Filter documents based on metadata key-value pair
    filtered_doc_ids = [
        doc_id for doc_id, doc in vector_store.docstore._dict.items()
        if doc.metadata.get("provider") == provider and doc.metadata.get("policy_name") == policy
    ]

    if not filtered_doc_ids:
        print("No documents match the given metadata filter.")
        return []
    
    # Step 2: Convert document IDs to FAISS indices
    faiss_indices = []
    for doc_id in filtered_doc_ids:
        if doc_id in vector_store.index_to_docstore_id.values():
            # Find the FAISS index that corresponds to this doc_id
            for idx, stored_id in vector_store.index_to_docstore_id.items():
                if stored_id == doc_id:
                    faiss_indices.append(idx)
                    break
    
    if not faiss_indices:
        print("Could not map document IDs to FAISS indices.")
        return []
    
    # Step 3: Get the query embedding
    query_embedding = embeddings.embed_query(query)
    
    # Step 4: Manually calculate distances to filtered documents only
    all_embeddings = vector_store.index.reconstruct_n(0, vector_store.index.ntotal)
    
    # Get embeddings for our filtered indices only
    filtered_embeddings = [all_embeddings[idx] for idx in faiss_indices]
    
    # Calculate similarities (using dot product or cosine similarity)
    similarities = []
    for idx, emb in zip(faiss_indices, filtered_embeddings):
        # Using dot product as similarity measure (higher is more similar)
        similarity = np.dot(query_embedding, emb)
        similarities.append((idx, similarity))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Take top k results
    top_k_indices = [idx for idx, _ in similarities[:k]]
    
    # Get the corresponding documents
    results = []
    for idx in top_k_indices:
        doc_id = vector_store.index_to_docstore_id[idx]
        doc = vector_store.docstore._dict[doc_id]
        results.append(doc)
    
    return results






    