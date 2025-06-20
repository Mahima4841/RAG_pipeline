import os
import requests
import fitz
from tqdm.auto import tqdm
import random
import pandas as pd
from spacy.lang.en import English
import re

from pdf_loader_and_chunk_generator import *
from embedding_model import *

def Get_embeddings():
    pdf_path = get_pdf_path()
    pages_and_texts = open_and_read_pdf(pdf_path)
    #print(pages_and_texts[:2])
    #show_df(pages_and_texts)

    sentencize_using_spacy(pages_and_texts)
    #show_df(pages_and_texts)

    split_pdf_to_chunks_of_sentences(pages_and_texts) # 6 sentence = 1 chunk
    #show_df(pages_and_texts)

    pages_and_chunks = split_each_chunk_to_own_item(pages_and_texts)
    # show_df(pages_and_chunks)
    # print(random.sample(pages_and_chunks, k=1))

    #show random chunks of token size <= 30
    pages_and_chunks_trimmed = show_and_omit_chunks_with_less_tokens(pages_and_chunks)
    print(pages_and_chunks_trimmed[:2])

    embeddings = convert_chunks_to_embeddings(pages_and_chunks_trimmed)
    print(embeddings[0])
    return embeddings, pages_and_chunks
