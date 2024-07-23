import pdfplumber
import streamlit as st
from pinecone import ServerlessSpec
from pinecone.grpc import PineconeGRPC as Pinecone
from transformers import AutoTokenizer, AutoModel

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

pdf_path = '/Users/akdiwaka/Downloads/PA_Checklist_Automation_guide.pdf'
output_path = 'output.txt'

# Initialize Pinecone
pc = Pinecone(api_key='7221a3bd-0cad-4034-abd5-a8c9022a4026')

index_name = 'pdf-index'
# Create an index if it doesn't already exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# Connect to the index
index = pc.Index(index_name)


def extract_text_from_pdf(_pdf_path):
    structured_data = []
    with pdfplumber.open(_pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text(layout=True)
            structured_data.append({
                'page': page_num,
                'text': text
            })
    return structured_data


def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embeddings.flatten().tolist()


# Store embeddings with metadata
def store_embeddings(_embeddings, _segments, _page_num):
    # breakpoint()
    for i, embedding in enumerate(_embeddings):
        vector = {
            'id': f'page_{_page_num}_segment_{i}',
            'values': embedding,
            'metadata': {'text': _segments[i], 'page': _page_num}
        }
        index.upsert(vectors=[vector])


def save_text_to_file(text, _output_path):
    with open(_output_path, 'w') as file:
        file.write(text)


structured_text = extract_text_from_pdf(pdf_path)
for data in structured_text:
    # Split text into chunks
    segments = [data['text'][i:i + 512] for i in range(0, len(data['text']), 512)]
    embeddings = [embed_text(segment) for segment in segments]
    store_embeddings(embeddings, segments, data['page'])

save_text_to_file(structured_text, output_path)
print(f"Extracted text saved to {output_path}")


def query_vector_db(_query, top_k=5):
    query_embedding = embed_text(_query)
    print(query_embedding)
    _results = index.query(vector=query_embedding, include_metadata=True, top_k=top_k)
    return _results


# Streamlit Interface
st.title("PDF Processing with Pinecone and Transformers")
# Example query
# query = "Get me command line to run pa check automation?"
query = st.text_input("Enter your query:")
if query:
    with st.status("Getting your data...", expanded=True) as status1:
        st.write("Searching for data...")
        results = query_vector_db(query)
        for result in results['matches']:
            st.write(f"Match: {result['metadata']['text']} (Score: {result['score']})")
        status1.update(label="Download complete!", state="complete", expanded=False)
    st.button("Rerun")
