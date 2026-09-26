import shutil
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

from ingestion import load_and_split_documents
load_dotenv()
# Project paths

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data"
VECTORSTORE_PATH = BASE_DIR / "vectorstore"

# Create vector store

def create_vector_store():

    # 1. Load and split all documents
    chunks = load_and_split_documents(str(DATA_PATH))

    # 2. Use Gemini embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2"
    )

    # 3. Remove old vector store
    if VECTORSTORE_PATH.exists():
        shutil.rmtree(VECTORSTORE_PATH)
        print("Removed old vector store.")

    # 4. Create new Chroma vector database
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTORSTORE_PATH)
    )

    print(f"Stored {len(chunks)} chunks in ChromaDB.")

    return vector_store


if __name__ == "__main__":
    create_vector_store()