import os
import torch
from dotenv import load_dotenv
from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer

# Load environment variables first so they can override defaults
load_dotenv()

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_LOCAL_MODEL_DIR = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "models", "Qwen2.5-3B-Instruct")

def ensure_local_model(local_model_dir: str, model_id: str) -> str:
    """Ưu tiên model local. Chỉ tải từ HF khi bật ALLOW_HF_DOWNLOAD=1."""
    local_ok = os.path.exists(local_model_dir) and os.path.exists(os.path.join(local_model_dir, "config.json"))
    if local_ok:
        return local_model_dir

    if os.getenv("ALLOW_HF_DOWNLOAD", "0") != "1":
        raise FileNotFoundError(
            f"Không tìm thấy model local tại {local_model_dir}. "
            "Bạn đã tắt auto-download (ALLOW_HF_DOWNLOAD != 1)."
        )

    print(f"📥 Không tìm thấy model local, bắt đầu tải {model_id} về {local_model_dir}...")
    snapshot_download(
        repo_id=model_id,
        local_dir=local_model_dir,
        local_dir_use_symlinks=False,
        token=os.getenv("HF_TOKEN"),
    )
    return local_model_dir

def train() -> None:
    print("🚀 Bắt đầu quá trình Huấn luyện mô hình (Fine-Tuning QLoRA)...")
    
    # 1. Cấu hình đường dẫn
    model_id = os.getenv("GENERATOR_MODEL", DEFAULT_MODEL_ID)
    local_model_dir = os.getenv("LOCAL_MODEL_DIR", DEFAULT_LOCAL_MODEL_DIR)
    dataset_path = "data/datasets/sft_train.jsonl"
    output_dir = "models/adapters/qwen-notebooklm"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Lỗi: Không tìm thấy file dữ liệu tại {dataset_path}")
        print("Bạn cần tạo dataset trước (ví dụ chạy script build_dataset).")
        return

    try:
        local_model_dir = ensure_local_model(local_model_dir, model_id)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return

    # 2. Tải Tokenizer và cấu hình định dạng ChatML của Qwen
    print(f"📦 Đang tải Tokenizer từ local model: {local_model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(
        local_model_dir,
        local_files_only=True,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # 3. Cấu hình lượng tử hóa (Quantization 4-bit) để tiết kiệm RAM/VRAM
    print("🗜️ Đang cấu hình nén mô hình 4-bit (QLoRA)...")
    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16_supported else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    # 4. Tải mô hình Qwen
    print(f"🧠 Đang tải mô hình gốc từ local: {local_model_dir}...")
    device_map = os.getenv("DEVICE_MAP", "auto")
    model = AutoModelForCausalLM.from_pretrained(
        local_model_dir,
        quantization_config=bnb_config,
        device_map=device_map,
        dtype=compute_dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    # 5. Cấu hình LoRA (Chỉ huấn luyện ~1% trọng số mới, giữ nguyên 99% kiến thức cũ)
    print("🔧 Đang gắn Adapter LoRA vào mô hình...")
    lora_config = LoraConfig(
        r=16, # Độ phân giải học tập
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # Không gắn LoRA thủ công ở đây; SFTTrainer sẽ nhận peft_config và gắn adapter.

    # 6. Tải dữ liệu huấn luyện
    print("📚 Đang nạp dữ liệu từ sft_train.jsonl...")
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    # Hàm chuyển đổi dữ liệu thành định dạng hội thoại cho SFTTrainer
    def formatting_prompts_func(example):
        # Trích xuất mảng "messages" từ file JSONL
        return tokenizer.apply_chat_template(example["messages"], tokenize=False)

    # 7. Cấu hình thông số huấn luyện
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,       # Giảm xuống 1 nếu máy báo hết VRAM (OOM)
        gradient_accumulation_steps=4,       # Tăng lên nếu giảm batch_size
        learning_rate=2e-4,
        logging_steps=5,
        max_steps=200,                       # Chạy thử 200 bước trước. Khi thật sự train có thể đổi thành num_train_epochs=3
        save_steps=50,
        fp16=not bf16_supported,             # Dùng fp16 nếu GPU không hỗ trợ bf16
        bf16=bf16_supported,
        optim="paged_adamw_8bit",
        report_to="none"                     # Tắt wandb nếu bạn chưa cấu hình
    )

    # 8. Khởi tạo Trainer và BẮT ĐẦU!
    print("🔥 BẮT ĐẦU HUẤN LUYỆN...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
        args=training_args,
        formatting_func=formatting_prompts_func,
    )

    trainer.model.print_trainable_parameters()

    trainer.train()

    # 9. Lưu thành quả
    print(f"💾 Đang lưu mô hình (Adapter) đã huấn luyện tại: {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("🎉 QUÁ TRÌNH HUẤN LUYỆN ĐÃ HOÀN TẤT THÀNH CÔNG!")

if __name__ == "__main__":
    train()