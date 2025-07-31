# Building clusters on the embedding space of the Bible

1. Download the bible
2. Embedd chunks
3. Cluster
4. Use LLM to assign summaries of each cluster 

## Setup

1. Install Python 3.10 or higher
2. Install the required packages using pip:
```bash
pip install -r requirements.txt
```
3. Set environment variables:
```bash
export PINECONE_API_KEY=YOUR_PINECONE_API_KEY
export OPENAI_API_KEY=YOUR_OPENAI_API_KEY
```
4. Run the following scripts in order:
```bash
python 01_upload.py
python 02_cluster.py
python 03_query.py
```
5. The script will output the answer to the user question.
