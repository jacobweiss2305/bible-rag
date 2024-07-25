
import os
import time
from typing import Dict, List, Tuple

from openai import OpenAI

from pinecone import Pinecone, ServerlessSpec

from tqdm import tqdm

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

logger.info("Starting initialization of Pinecone.")


from dotenv import load_dotenv

load_dotenv()


# Initialize OpenAI client
client = OpenAI()

def initialize_pinecone(index_name: str, dimension: int = 1536):
    """Initialize Pinecone and create an index if it doesn't exist."""
    logging.basicConfig(level=logging.INFO)

    logging.info(f"Initializing Pinecone with index name: {index_name} and dimension: {dimension}")
    spec = ServerlessSpec(cloud="aws", region="us-east-1")
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

    if index_name not in pc.list_indexes().names():
        logging.info(f"Index {index_name} does not exist. Creating new index.")
        pc.create_index(index_name, dimension=dimension, spec=spec)
        while not pc.describe_index(index_name).status["ready"]:
            logging.info(f"Waiting for index {index_name} to be ready...")
            time.sleep(1)
    else:
        logging.info(f"Index {index_name} already exists.")

    index = pc.Index(index_name)
    time.sleep(1)
    logging.info(f"Index {index_name} is ready and initialized.")
    return index

def embed_text(text: str) -> List[float]:
    """Generate embedding for a given text."""
    response = client.embeddings.create(
        model="text-embedding-ada-002", input=text[:8000]
    )
    return response.data[0].embedding

def upsert_ticket(index, id: str, text_to_embed: str, metadata: Dict):
    embedding = embed_text(text_to_embed)
    vectors = [{"id": id, "values": embedding, "metadata": metadata}]
    index.upsert(vectors)
    return embedding

def query(index, id: str, top_k: int = 1) -> List[Dict]:
    """
    Query the index for the given query text.
    """
    result = index.query(
        vector=[0] * 1536,  # Dummy vector for metadata-only query
        filter={"id": {"$eq": id}},
        top_k=top_k,
        include_metadata=True,
    )
    return result

def fetch_all_ids(index):
    """Fetch all ids from the index."""
    return [i for i in index.list()]
