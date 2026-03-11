from fastapi import APIRouter

router = APIRouter()


@router.post("/")
def trigger_train() -> dict:
    return {"status": "queued-placeholder"}
