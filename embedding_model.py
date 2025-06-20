from sentence_transformers import SentenceTransformer
import numpy
import torch
import pandas as pd
import chromadb
from chromadb.config import Settings
from tqdm.auto import tqdm

embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device="cpu")

def random_try():
    #sentences = ["This a dog.", "She is good at Maths.", "Do your work.", "I hope RCB wins this time."]
    sentences = ["In this case, the arg variable is not altered in place.", "It seems that Python treats your supplied argument as a standalone value rather than a reference to an existing variable.", "Does this mean Python passes arguments by value rather than by reference?",
                "Not quite. Python passes arguments neither by reference nor by value, but by assignment.",  "Below, you’ll quickly explore the details of passing by value and passing by reference before looking more closely at Python’s approach.", "After that, you’ll walk through some best practices for achieving the equivalent of passing by reference in Python."
                ]
    embeddings = embedding_model.encode(sentences)
    embedding_dict = dict(zip(sentences, embeddings))

    for sentence, embedding in embedding_dict.items():
        print("Sentence: ", sentence)
        print("embedding: ", embedding)
        print("")

def convert_chunks_to_embeddings(pages_and_chunks_trimmed: list[dict]):

    for item in tqdm(pages_and_chunks_trimmed):
        item["embedding"] = embedding_model.encode(item["joined_sentence_chunk"])

    text_chunk = [item["joined_sentence_chunk"] for item in pages_and_chunks_trimmed]
    text_chunk_embeddings = embedding_model.encode(text_chunk, batch_size=32, convert_to_tensor=True)
    # text_chunk_embeddings_dict = dict(zip(text_chunk, text_chunk_embeddings))
    # for sentence, embedding in text_chunk_embeddings_dict.items():
    #     print("Sentence: ", sentence)
    #     print("embedding: ", embedding)
    #     print("")

    #store embeddings in chroma db
    # chroma_client = chromadb.PersistentClient(path="./chroma_db")
    # collection = chroma_client.get_or_create_collection(name="pdf_embeddings")

    # ids = [f"chunk_{i}" for i in range(len(text_chunk))]

    # collection.add(
    #     documents=text_chunk,
    #     embeddings=text_chunk_embeddings,
    #     ids=ids
    # )

    #other option is to save it in a csv file
    text_chunk_embeddings_df = pd.DataFrame(pages_and_chunks_trimmed)
    embeddings_df_save_path = "text_chunk_embeddings_df.csv"
    text_chunk_embeddings_df.to_csv(embeddings_df_save_path, index=False)

    text_chunk_embeddings_df_load = pd.read_csv(embeddings_df_save_path)
    # print(text_chunk_embeddings_df_load.head())
    return text_chunk_embeddings



   