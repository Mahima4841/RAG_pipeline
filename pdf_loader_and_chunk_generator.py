import os
import requests
import fitz
from tqdm.auto import tqdm
import random
import pandas as pd
from spacy.lang.en import English
import re

#get pdf path
def get_pdf_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(script_dir, "UFS.pdf")

    if not os.path.exists(pdf_path):
        print("Failed to retrieve pdf ufs.pdf")
    else:
        print("Pdf path exists")

    return pdf_path

#get all text of each page in single line
def text_formatter(text: str) -> str:
    cleaned_text = text.replace("\n", " ").strip()
    return cleaned_text

def open_and_read_pdf(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages_and_texts = []
    for page_number, page in tqdm(enumerate(doc)):
        text = page.get_text() #plain text as UTF-8
        text = text_formatter(text)
        pages_and_texts.append({"page_number": page_number - 20, #pdf content starts from page 19
                                "page_char_count": len(text),
                                "page_word_count": len(text.split(" ")),
                                "page_sentence_count_raw": len(text.split(". ")),
                                "page_token_count": len(text) / 4, #taking 1 token ~ 4chars
                                "text": text
                                })
        
    return pages_and_texts

def sentencize_using_spacy(pages_and_texts: list[dict]):
    #define nlp ---> can be used for large texts as a sentencizing pipeline
    nlp = English()
    nlp.add_pipe("sentencizer")

    #define sentencizing pipeline suitable for usecase like single pdf
    for item in tqdm(pages_and_texts): #will take each page one by one and convert to sentences
        item["sentences"] = list(nlp(item["text"]).sents)

        item["sentences"] = [str(sentence) for sentence in item["sentences"]] #to make sure all sentences are in string format

        item["page_sentence_count_spacy"] = len(item["sentences"])

#function to split input text into sublists of desired size
def split_list(input_list: list, slice_size: int) -> list[list[str]]:
    return [input_list[i:i+slice_size] for i in range(0, len(input_list), slice_size)]

def split_pdf_to_chunks_of_sentences(pages_and_texts: list[dict]):
    chunk_size = 6 #no. of sentences you want to keep in one chunk. Here, 1 chunk = 6 sentences
    for item in tqdm(pages_and_texts):
        item["sentence_chunks"] = split_list(input_list=item["sentences"], slice_size=chunk_size)
        item["num_chunks"] = len(item["sentence_chunks"])

def show_df(pages_and_texts: list[dict]):
    df = pd.DataFrame(pages_and_texts)
    print(df.describe().round(2)) #to get stat of data like mean, no. of pages, no. of words, etc.

def split_each_chunk_to_own_item(pages_and_texts: list[dict]) -> list[dict]:
    pages_and_chunks = []
    for item in tqdm(pages_and_texts):
        for sentence_chunk in item["sentence_chunks"]:
            chunk_dict = {}
            chunk_dict["page_number"] = item["page_number"] 

            #join each sentence of chunk into single paragraph like structure
            joined_sentence_chunk = "".join(sentence_chunk).replace("  "," ").strip()
            joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk)  # ".A" -> ". A" for any full-stop/capital letter combo 
            chunk_dict["joined_sentence_chunk"] = joined_sentence_chunk

            #get stats
            chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
            chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
            chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4 # 1 token ~ 4 chars

            pages_and_chunks.append(chunk_dict)

    print("We have {} chunks".format(len(chunk_dict)))
    return pages_and_chunks

def show_and_omit_chunks_with_less_tokens(pages_and_chunks: list[dict]) -> list[dict]:
    min_token_length = 30
    df = pd.DataFrame(pages_and_chunks)
    for row in df[df["chunk_token_count"] <= min_token_length].sample(5).iterrows():
        print(f'Chunk token count: {row[1]["chunk_token_count"]} | Text: {row[1]["joined_sentence_chunk"]}')

    pages_and_chunks_trimmed = df[df["chunk_token_count"] > min_token_length].to_dict(orient="records")
    return pages_and_chunks_trimmed


