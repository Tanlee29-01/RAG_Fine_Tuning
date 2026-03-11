from fastapi import APIRouter
from pydantic import BaseModel
from src.api.rag_service import run_rag

router = APIRouter()

class ChatRequest(BaseModel):
    question: str

@router.post("/")
def chat(payload: ChatRequest) -> dict:
    # Gọi trực tiếp luồng RAG
    result = run_rag(payload.question)
    return result