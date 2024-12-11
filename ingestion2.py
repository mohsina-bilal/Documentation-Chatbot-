import os
from dotenv import load_dotenv
load_dotenv()
INDEX_NAME = "langchain-doc-index-2"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

from langchain.text_splitter import RecursiveCharacterTextSplitter # to chunkify data
from langchain_community.document_loaders.readthedocs import ReadTheDocsLoader # helps to build github repositories
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# Initialize Pinecone with the new method
pc = Pinecone(api_key=PINECONE_API_KEY, environment="us-east-1")

embeddings = OpenAIEmbeddings(model = "text-embedding-3-large") # can check with different embedding models

def ingest_docs():

    docs_path = r"C:\Users\mohsi\PycharmProjects\Documentation_Helper\documentation-helper\langchain-docs\langchain-docs\api.python.langchain.com\en\latest"
    loader = ReadTheDocsLoader(path = docs_path, encoding = "utf-8")

    raw_documents = []
    error_files = []

    # Attempt to lazy load all documents, catching errors for individual files
    try:
        for doc in loader.lazy_load():
            raw_documents.append(doc)
    except Exception as e:
        print(f"An unexpected error occurred while loading documents: {e}")
        return

    print(f"Loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 600, chunk_overlap = 50)

    documents = []

    # Attempt to split documents and handle errors
    for doc in raw_documents:
        try:
            # If document is in the expected format, split it
            chunks = text_splitter.split_documents([doc])
            documents.extend(chunks)
        except Exception as e:
            print(f"Error processing document: {e}")
            error_files.append(str(e))

    # Update document metadata for URL correction
    for doc in documents:
        if "source" in doc.metadata:
            new_url = doc.metadata["source"].replace("langchain-docs", "https:/")
            doc.metadata.update({"source": new_url})

    print(f"Going to add {len(documents)} to Pinecone")

    PineconeVectorStore.from_documents(
        documents, embeddings, index_name=INDEX_NAME
    )

    print("----- Loading to VectorStore done -----")

if __name__ == "__main__":
    ingest_docs()