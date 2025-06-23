import os
import requests
import fitz
from tqdm.auto import tqdm
import random
import pandas as pd
from spacy.lang.en import English
import re
import chromadb
from chromadb.config import Settings
import numpy as np

from pdf_loader_and_chunk_generator import *
from embedding_model import *

def generate_and_store_embeddings():
    """
    Generate embeddings from PDF and store them in ChromaDB and CSV
    """
    print("Starting embedding generation process...")
    
    pdf_path = get_pdf_path()
    pages_and_texts = open_and_read_pdf(pdf_path)
    print("PDF loaded and processed")

    sentencize_using_spacy(pages_and_texts)
    print("Text sentencized using spaCy")

    split_pdf_to_chunks_of_sentences(pages_and_texts) # 6 sentence = 1 chunk
    print("PDF split into chunks")

    pages_and_chunks = split_each_chunk_to_own_item(pages_and_texts)
    print(f"Total chunks created: {len(pages_and_chunks)}")

    # Show random chunks of token size <= 30
    pages_and_chunks_trimmed = show_and_omit_chunks_with_less_tokens(pages_and_chunks)
    print(f"Chunks after trimming: {len(pages_and_chunks_trimmed)}")

    # Generate embeddings
    embeddings, pages_and_chunks_trimmed = convert_chunks_to_embeddings(pages_and_chunks_trimmed)
    print("Embeddings generated successfully")
    
    # Store in ChromaDB
    store_in_chromadb(pages_and_chunks_trimmed, embeddings)
    print("Embeddings stored in ChromaDB")
    
    return embeddings, pages_and_chunks_trimmed

def store_in_chromadb(pages_and_chunks_trimmed, embeddings):
    """
    Store embeddings in ChromaDB
    """
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="pdf_embeddings")
    
    # Clear existing data - use proper syntax
    try:
        collection.delete(where={"$and": []})
    except:
        pass
    
    text_chunks = [item["joined_sentence_chunk"] for item in pages_and_chunks_trimmed]
    ids = [f"chunk_{i}" for i in range(len(text_chunks))]
    
    # Ensure embeddings is a list of lists (not tensor or numpy array)
    if hasattr(embeddings, 'cpu'):
        embeddings_list = embeddings.cpu().numpy().tolist()
    elif isinstance(embeddings, np.ndarray):
        embeddings_list = embeddings.tolist()
    else:
        embeddings_list = list(embeddings)
    # Ensure the number of embeddings matches the number of documents
    if len(embeddings_list) != len(text_chunks):
        raise ValueError(f"Number of embeddings ({len(embeddings_list)}) does not match number of documents ({len(text_chunks)})")
    collection.add(
        documents=text_chunks,
        embeddings=embeddings_list,
        ids=ids
    )
    print(f"Stored {len(text_chunks)} embeddings in ChromaDB")

if __name__ == "__main__":
    generate_and_store_embeddings()
    print("Embedding generation completed!") 