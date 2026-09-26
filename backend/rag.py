import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
VECTORSTORE_PATH = BASE_DIR / "vectorstore"


# 1. Load embedding model

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2"
)

# 2. Load existing vector DB

vector_store = Chroma(
    persist_directory=str(VECTORSTORE_PATH),
    embedding_function=embeddings
)


# 3. Create retriever

retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)

# 4. Create LLM

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0
)


# 5. Create prompt

prompt = ChatPromptTemplate.from_template(
    """
You are an AI assistant that answers questions using the
provided company documents.

Rules:
1. Answer only using the provided context.
2. If the answer is not present in the context, say:
   "I could not find this information in the provided documents."
3. Do not make up information.
4. Keep the answer clear and concise.

Context:
{context}

Question:
{question}

Answer:
"""
)


def ask_question(question: str):

    # Retrieve relevant documents
    documents = retriever.invoke(question)

    # Combine retrieved chunks
    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    # Create prompt
    formatted_prompt = prompt.invoke({
        "context": context,
        "question": question
    })

    # Ask LLM
    response = llm.invoke(formatted_prompt)

    return response.text, documents


if __name__ == "__main__":

    question = "What security measures are required for company laptops?"

    answer, documents = ask_question(question)

    print("\nANSWER:")
    print(answer)

    print("\nSOURCES:")

    for document in documents:
        print(document.metadata)