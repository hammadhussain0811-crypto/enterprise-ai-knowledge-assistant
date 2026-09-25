from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_split_documents(data_dir: str = "data"):
    """
    Load all PDF documents from the data directory
    and split them into smaller chunks.
    """

    data_path = Path(data_dir)

    all_documents = []

    # Find all PDF files
    pdf_files = list(data_path.glob("*.pdf"))

    print(f"Found {len(pdf_files)} PDF files.")

    # Load every PDF
    for pdf_file in pdf_files:

        print(f"Loading: {pdf_file.name}")

        loader = PyPDFLoader(str(pdf_file))
        documents = loader.load()

        all_documents.extend(documents)

    print(f"Loaded {len(all_documents)} pages in total.")

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(all_documents)

    print(f"Created {len(chunks)} chunks.")

    return chunks