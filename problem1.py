import pinecone
from transformers import AutoTokenizer, AutoModel
import torch
import openai

pinecone.init(api_key='fa1bce38-f58a-488f-8ffe-189412abcd19', environment='us-west1-gcp')
index_name = 'document-embedding-index'
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=768)
index = pinecone.Index(index_name)
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    embedding = torch.mean(outputs.last_hidden_state, dim=1)
    return embedding.detach().numpy()[0]


documents = [
    {"id": "doc1", "text": "This is a document about business strategies."},
    {"id": "doc2", "text": "Another document discussing the future of AI."}
]

for doc in documents:
    embedding = get_embedding(doc["text"])
    index.upsert([(doc["id"], embedding)])


def retrieve_relevant_documents(query):
    query_embedding = get_embedding(query)
    result = index.query(query_embedding, top_k=5, include_metadata=True)
    return result['matches']


query = "What are some business strategies?"
relevant_docs = retrieve_relevant_documents(query)
for doc in relevant_docs:
    print(f"Document ID: {doc['id']}, Score: {doc['score']}")

openai.api_key = 'fa1bce38-f58a-488f-8ffe-189412abcd19'


def generate_answer_from_docs(query, relevant_docs):
    context = " ".join([doc['metadata']['text'] for doc in relevant_docs])
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )

    return response.choices[0].text.strip()


# Generate answer
answer = generate_answer_from_docs(query, relevant_docs)
# print(f"Answer: {answer}")
test_queries = [
    "Tell me about AI in the future.",
    "What are the best business strategies?"
]

for query in test_queries:
    relevant_docs = retrieve_relevant_documents(query)
    answer = generate_answer_from_docs(query, relevant_docs)
    print(f"Query: {query}\nAnswer: {answer}\n")