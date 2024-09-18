import cohere
import pinecone
import PyPDF2
import streamlit as st
from sentence_transformers import SentenceTransformer

# Initialize Cohere and Pinecone
cohere_api_key = "fa1bce38-f58a-488f-8ffe-189412abcd19"
co = cohere.Client(cohere_api_key)

pinecone_api_key = "YOUR_PINECONE_API_KEY"
pinecone.init(api_key=pinecone_api_key, environment="fa1bce38-f58a-488f-8ffe-189412abcd19")

index_name = "document-embeddings"
index = pinecone.Index(index_name)

# Load sentence transformer model for embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in range(len(reader.pages)):
        text += reader.pages[page].extract_text()
    return text

# Function to split text into smaller chunks
def split_text(text, chunk_size=500):
    sentences = text.split('. ')
    chunks, chunk = [], []
    for sentence in sentences:
        chunk.append(sentence)
        if len(' '.join(chunk)) > chunk_size:
            chunks.append(' '.join(chunk))
            chunk = []
    if chunk:
        chunks.append(' '.join(chunk))
    return chunks

# Function to generate embeddings and store in Pinecone
def generate_embeddings_and_store(text_chunks):
    for i, chunk in enumerate(text_chunks):
        embedding = embedder.encode(chunk).tolist()
        index.upsert([(str(i), embedding)])

# Function to retrieve the most relevant document chunk
def retrieve_relevant_chunk(query, top_k=1):
    query_embedding = embedder.encode([query]).tolist()[0]
    results = index.query([query_embedding], top_k=top_k)
    return [result['metadata']['text'] for result in results['matches']]

# Function to generate answer from Cohere using retrieved chunks
def generate_answer(relevant_chunks, query):
    context = " ".join(relevant_chunks)
    response = co.generate(
        model='command-xlarge-nightly',
        prompt=f"Context: {context}\nQuestion: {query}\nAnswer:",
        max_tokens=100
    )
    return response.generations[0].text.strip()
# Initialize Pinecone index if it doesn't exist
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=384)

# Streamlit interface
st.title("Document QA Bot")
st.write("Upload a PDF and ask questions based on its content.")

# PDF upload
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Extract text from PDF
    text = extract_text_from_pdf(uploaded_file)
    st.write("Document uploaded and processed successfully!")

    # Split text into chunks
    text_chunks = split_text(text)
    st.write(f"Document split into {len(text_chunks)} chunks for processing.")

    # Generate embeddings and store in Pinecone
    generate_embeddings_and_store(text_chunks)
    st.write("Document embeddings generated and stored in the vector database.")

    # Query input
    query = st.text_input("Ask a question based on the document content:")

    if query:
        # Retrieve relevant document chunks
        relevant_chunks = retrieve_relevant_chunk(query)
        st.write("Retrieved document segments:")
        for chunk in relevant_chunks:
            st.write(chunk)

        # Generate and display answer
        answer = generate_answer(relevant_chunks, query)
        st.write(f"Answer: {answer}")

