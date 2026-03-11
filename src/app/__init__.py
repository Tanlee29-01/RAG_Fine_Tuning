# Backward-compatibility shim: src.app -> src.api
from src.api.api import create_app  # noqa: F401