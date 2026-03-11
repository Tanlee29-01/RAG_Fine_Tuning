import hashlib


def md5_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()
