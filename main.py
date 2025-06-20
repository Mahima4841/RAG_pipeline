import os
import requests
import fitz
from tqdm.auto import tqdm
import random
import pandas as pd
from spacy.lang.en import English
import re
import textwrap
import chromadb

from time import perf_counter as timer
from pdf_loader_and_chunk_generator import *
from embedding_model import *
from Get_embeddings import *
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

import ollama
 
embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device="cpu")

embeddings, pages_and_chunks = Get_embeddings()

query = "What is RPMB region in UFS?"
print(f"Query: {query}")

query_embedding = embedding_model.encode(query, convert_to_tensor=True)

dot_scores = util.dot_score(a=query_embedding, b=embeddings)

top_results_dot_product = torch.topk(dot_scores, k=1)
print(top_results_dot_product)

def print_wrapped(text, wrap_length=80):
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)
    return wrapped_text

print(f"Results: ")

for score, idx in zip(top_results_dot_product[0][0], top_results_dot_product[1][0]):
    print(f"Score: {score}")
    print("Text: ")
    context = print_wrapped(pages_and_chunks[idx]["joined_sentence_chunk"])
    print(f"Page number: {pages_and_chunks[idx]['page_number']}")
    print("\n")

# model_id = "C:/Users/MAHIMAK/AppData/Local/Ollama"

# tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id)

# llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id, 
                                                #  torch_dtype=torch.float16
                                                #  )


response = ollama.generate(
    model="mistral",
    prompt=f"Context: {context}\n\nQuestion: {query}\n\nAnswer:",
    options={
        "num_predict": 128,
        "temperature": 0.3,
        "top_k": 20
    }
)

print(response["response"])

# chroma_client = chromadb.PersistentClient(path="./chroma_db")
# collection = chroma_client.get_collection(name="pdf_embeddings")
# all = collection.get()
# print(all.keys())
# print(all['embeddings'][0])

