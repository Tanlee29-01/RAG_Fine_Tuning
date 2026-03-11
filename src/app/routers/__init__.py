# Backward-compatibility shim: src.app.routers -> src.api.routers
from src.api.routers import chat, documents, health, train  # noqa: F401