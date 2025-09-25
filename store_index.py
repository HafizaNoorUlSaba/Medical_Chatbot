from dotenv import load_dotenv
import os
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, download_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()


PINECONE_API_KEY= os.getenv("PINECONE_API_KEY")
GROQ_API_KEY= os.getenv("GROQ_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

extracted_data = load_pdf_files(r"D:\nr\Medical_chatbot\Medical_Chatbot\data")
filter_data = filter_to_minimal_docs(extracted_data)
texts_chunks = text_split(filter_data)
embedding = download_embeddings()

pinecone_api_key=PINECONE_API_KEY
pc=Pinecone(api_key=pinecone_api_key)

index_name="medical-chatbot"
# Create index if not exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,   # must match your embedding size
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
index = pc.Index(index_name)

docsearch = PineconeVectorStore.from_documents(
    documents=texts_chunks,
    embedding=embedding,
    index=index,
    namespace="medical-chatbot"
) 