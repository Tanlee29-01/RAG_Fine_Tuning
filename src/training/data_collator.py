class SimpleDataCollator:
    def __call__(self, features: list[dict]) -> dict:
        return {"features": features}
