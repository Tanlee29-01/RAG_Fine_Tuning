# Backward-compatibility shim: src.llm.adapters -> src.generation.adapters
from src.generation.adapters.qwen_lora_loader import load_qwen_lora  # noqa: F401