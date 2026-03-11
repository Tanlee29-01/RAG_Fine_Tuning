"""Entry point for QLoRA fine-tuning. Reads config from .env / configs/train.yaml."""
from src.finetuning.train_qlora import train


if __name__ == "__main__":
    train()
