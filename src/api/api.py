from fastapi import FastAPI
from src.api.routers import chat, documents, train, health


def create_app() -> FastAPI:
    app = FastAPI(title="RAG Fine-Tuning API")
    app.include_router(health.router)
    app.include_router(chat.router, prefix="/chat", tags=["chat"])
    app.include_router(documents.router, prefix="/documents", tags=["documents"])
    app.include_router(train.router, prefix="/train", tags=["train"])
    return app
