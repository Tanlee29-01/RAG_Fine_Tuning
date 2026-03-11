def seed_everything(seed: int = 42) -> None:
    try:
        import random
        random.seed(seed)
    except Exception:
        pass
