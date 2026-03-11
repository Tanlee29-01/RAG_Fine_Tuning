from fastapi import APIRouter

router = APIRouter()


@router.get("/")
def list_documents() -> dict:
    return {"items": []}
