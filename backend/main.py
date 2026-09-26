from fastapi import FastAPI
from pydantic import BaseModel

from backend.rag import ask_question


app = FastAPI(
    title="Enterprise AI Knowledge Assistant",
    description="RAG-based API for querying company documents",
    version="1.0.0"
)


class QuestionRequest(BaseModel):
    question: str


class QuestionResponse(BaseModel):
    answer: str
    sources: list


@app.get("/")
def root():
    return {
        "message": "Enterprise AI Knowledge Assistant API is running"
    }


@app.post("/api/chat", response_model=QuestionResponse)
def chat(request: QuestionRequest):

    answer, documents = ask_question(request.question)

    sources = []

    for document in documents:
        sources.append({
            "source": document.metadata.get("source"),
            "page": document.metadata.get("page", 0) + 1
        })

    return {
        "answer": answer,
        "sources": sources
    }