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
from vectordb import fetch_or_insert_question
from tqdm import tqdm
import pandas as pd

if "data.csv" not in os.listdir():
    embeddings = []
    texts = []
    ids = []
    for i, line in tqdm(enumerate(lines), total=len(lines), desc="Fetching Embeddings"):
        id = str(i)
        metadata = {"id": id, "text": line}
        text, embedding = fetch_or_insert_question(index, id, line, metadata)
        embeddings.append(embedding)
        texts.append(text)
        ids.append(id)
    df = pd.DataFrame(embeddings, index=ids)
    df.index.name = 'id'
    df.to_csv("data.csv")
else:
    df = pd.read_csv("data.csv", index_col='id')

# Step 4: Normalize and Cluster
from cluster.main import normalize_X, cluster_kmeans_embeddings, find_optimal_clusters

embeddings = df.to_numpy()
X_normalized = normalize_X(embeddings)

# Find optimal number of clusters
max_k = 15
optimal_k = find_optimal_clusters(X_normalized, max_k)

# hardcoded for now
optimal_k = 15
labels = cluster_kmeans_embeddings(X_normalized, 15)
df["cluster"] = labels
df.to_csv("data.csv",)

# Plot historgram of clusters
from matplotlib import pyplot as plt
df["cluster"].value_counts(normalize = True).plot(kind='barh', figsize=(12, 5))
plt.show()


for idx, row in tqdm(df.iterrows(), total=len(df), desc="Updating Pinecone metadata"):
    index.update(
        id=str(idx),  # Convert index to string as Pinecone requires string ids
        set_metadata={"cluster": int(row['cluster'])},  # Convert cluster to int for consistency
    )

print("Metadata update complete.")