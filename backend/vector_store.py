import shutil
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from ingestion import load_and_split_documents


VECTORSTORE_PATH = "vectorstore"


def create_vector_store():

    # 1. Loading and spliting all documents
    chunks = load_and_split_documents("data")

    # 2. Loading embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 3. Removing old vector store
    if Path(VECTORSTORE_PATH).exists():
        shutil.rmtree(VECTORSTORE_PATH)
        print("Removed old vector store.")

    # 4. Creating new Chroma vector database
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTORSTORE_PATH
    )

    print(f"Stored {len(chunks)} chunks in ChromaDB.")

    return vector_store


if __name__ == "__main__":
    create_vector_store()