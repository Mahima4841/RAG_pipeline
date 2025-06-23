import os
import chromadb
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import torch
import ollama
import textwrap
import numpy as np

def load_embeddings_from_chromadb():
    """
    Load embeddings from ChromaDB
    """
    try:
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        collection = chroma_client.get_collection(name="pdf_embeddings")
        
        # Get all data from collection, explicitly including embeddings
        results = collection.get(include=['embeddings', 'documents'])
        print("DEBUG: ChromaDB collection.get() results:", {k: type(v) for k, v in results.items()})
        print("DEBUG: Number of documents:", len(results.get('documents', [])))
        print("DEBUG: Number of embeddings:", len(results.get('embeddings', [])))
        
        if not results['documents']:
            raise Exception("No embeddings found in ChromaDB. Please run generate_embeddings.py first.")
        
        return results['documents'], results['embeddings']
    
    except Exception as e:
        print(f"Error loading from ChromaDB: {e}")
        print("Please run generate_embeddings.py first to create embeddings.")
        return None, None

def query_rag(query_text, top_k=1):
    """
    Query the RAG system
    """
    print(f"Query: {query_text}")
    
    # Load embeddings from ChromaDB
    documents, embeddings = load_embeddings_from_chromadb()
    
    if documents is None:
        return None
    
    # Initialize embedding model
    embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device="cpu")
    
    # Encode query
    query_embedding = embedding_model.encode(query_text, convert_to_tensor=True)
    
    # Convert embeddings to tensor if they're not already
    if isinstance(embeddings, np.ndarray):
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    elif not isinstance(embeddings, torch.Tensor):
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    else:
        embeddings_tensor = embeddings.float()
    
    # Calculate similarity scores
    dot_scores = util.dot_score(a=query_embedding, b=embeddings_tensor)
    
    # Get top results
    top_results_dot_product = torch.topk(dot_scores, k=top_k)
    
    return top_results_dot_product, documents

def print_wrapped(text, wrap_length=80):
    """
    Print text with wrapping
    """
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)
    return wrapped_text

def generate_response(context, query):
    """
    Generate response using Ollama
    """
    response = ollama.generate(
        model="llama2",
        prompt=f"Context: {context}\n\nQuestion: {query}\n\nAnswer:",
        options={
            "num_predict": 250,
            "temperature": 0.3,
            "top_k": 20
        }
    )
    return response["response"]

def main():
    """
    Main function to run the RAG query system
    """
    # Example query
    query = "What is RPMB region in UFS?"
    
    # Query the RAG system
    results = query_rag(query, top_k=1)
    
    if results is None:
        return
    
    top_results_dot_product, documents = results
    
    print(f"Results: ")
    
    for score, idx in zip(top_results_dot_product[0][0], top_results_dot_product[1][0]):
        print(f"Score: {score}")
        print("Text: ")
        context = print_wrapped(documents[idx])
        print("\n")
        
        # Generate response using Ollama
        response = generate_response(context, query)
        print("Generated Response:")
        print(response)

if __name__ == "__main__":
    main() 