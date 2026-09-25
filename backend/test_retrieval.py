from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


VECTORSTORE_PATH = "vectorstore"


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    persist_directory=VECTORSTORE_PATH,
    embedding_function=embeddings
)


query = "What security measures are required for company laptops?"

results = vector_store.similarity_search(query, k=4)


print("\nRelevant documents:\n")

for i, document in enumerate(results, start=1):

    print(f"--- Result {i} ---")

    print(document.page_content)

    print("\nMetadata:")
    print(document.metadata)

    print()