# Step 1: Download the text files
from download_data import download_text_file
import os

if "the_bible.txt" not in os.listdir():
    download_text_file("https://openbible.com/textfiles/asv.txt", "the_bible.txt")

with open("the_bible.txt", "r") as file:
    text = file.read()

lines = text.split('\n')

# Step 2: Initialize Pinecone
from vectordb import initialize_pinecone

index = initialize_pinecone("bible-rag")

# Step 3: Fetch or Insert Text
from vectordb import upsert_ticket
from tqdm import tqdm

for i, line in tqdm(enumerate(lines), total=len(lines), desc="Fetch or Upsert Embeddings"):
        id = str(i)
        metadata = {"id": id, "text": line, }
        upsert_ticket(index, id, line, metadata)