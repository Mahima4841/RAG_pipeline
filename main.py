import os
import fitz
from tqdm.auto import tqdm
import random
import pandas as pd
from spacy.lang.en import English
import re
import textwrap
import chromadb

from time import perf_counter as timer
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

import ollama

# Import the query module
from query_rag import query_rag, print_wrapped, generate_response

def main():
    """
    Main function to run RAG queries
    """
    # Example query - you can modify this or make it interactive
    query = "What is RPMB region in UFS?"
    
    print("RAG Query System")
    print("=" * 50)
    
    # Query the RAG system
    results = query_rag(query, top_k=4)
    
    if results is None:
        print("Failed to load embeddings. Please run generate_embeddings.py first.")
        return
    
    # print(results)
    top_results_dot_product, documents = results
    
    print(f"Results: ")

    context = ""
    
    for score, idx in zip(top_results_dot_product[0][0], top_results_dot_product[1][0]):
        print(f"Score: {score}")
        print("Text: ")
        context += (" " + print_wrapped(documents[idx]))
        print("\n")
        
    # Generate response using Ollama
    response = generate_response(context, query)
    print("Generated Response:")
    print(response)

if __name__ == "__main__":
    main()

