# bible-rag
Question and answering using unsupervised cluster rag methods and Phidata Assistant on the bible

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
