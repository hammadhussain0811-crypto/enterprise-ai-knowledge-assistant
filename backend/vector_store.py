from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from ingestion import load_and_split_pdf


PDF_PATH = r"G:\enterprise-AI-knowledge-assistant\data\sample_company_handbook.pdf"
VECTORSTORE_PATH = "vectorstore"


def create_vector_store():

    # 1. Load and split PDF
    chunks = load_and_split_pdf(PDF_PATH)

    # 2. Load embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 3. Create Chroma vector database
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTORSTORE_PATH
    )

    print(f"Stored {len(chunks)} chunks in ChromaDB.")

    return vector_store


if __name__ == "__main__":
    create_vector_store()