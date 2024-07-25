# ask holisitc questions about the text

# Step 1: Initialize Pinecone
from vectordb import initialize_pinecone

index = initialize_pinecone("bible-rag")

# Step 2: Fetch unique cluster ids
import pandas as pd

clusters = pd.read_csv("data.csv",)
cluster_ids = clusters["cluster" ].unique()

# Step 3: Query the index for each cluster
from vectordb import query
set_sample_size = 10

sample_set = {}
for cluster_id in cluster_ids:
    filter_data_by_cluster = clusters[clusters["cluster"]==cluster_id].sample(set_sample_size)
    text_to_keep = []
    for id in filter_data_by_cluster["id"].values:
        text = query(index, str(id))["matches"][0]["metadata"]["text"]
        text_to_keep.append(text)
    text_that_needs_to_be_summarized = "\n".join(text_to_keep) 
    sample_set[str(cluster_id)] = text_that_needs_to_be_summarized

# Step 4: Summarize and Progress
from phi.assistant import Assistant
from phi.llm.openai import OpenAIChat

assistant = Assistant(
    llm=OpenAIChat(model="gpt-4o-mini"),
    instructions = ["Answer the users question given the context provided."],
)

answers = []
for cluster_id in cluster_ids:
    question = "User Question:\nlist all the relationships and show connections"
    context = "\nContext:\n" + sample_set[str(cluster_id)]
    full_text = question + context
    answer = assistant.run(full_text, stream=False)
    answers.append(answer)

# Step 5: Print the answers
question = "User Question:\nlist all the relationships and show connections"
full_answers = "\n".join(answers)
full_question = question + "Context: \n" + full_answers
result = assistant.run(full_answers, stream=False)
print(result)
