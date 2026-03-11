import os
import threading
from dotenv import load_dotenv

load_dotenv()

# Biến toàn cục để lưu model trong bộ nhớ, tránh tải lại nhiều lần
tokenizer = None
model = None
_lock = threading.Lock()

SYSTEM_PROMPT = """Bạn là một trợ lý AI phân tích tài liệu (giống NotebookLM). 
Nhiệm vụ của bạn là trả lời câu hỏi dựa TRÊN NGỮ CẢNH được cung cấp.
BẮT BUỘC phải trả về kết quả dưới dạng JSON theo đúng cấu trúc sau:
{
  "answer": "Câu trả lời chi tiết của bạn",
  "citations": [
    {
      "doc_id": "Tên_file_hoặc_ID_tài_liệu",
      "chunk_id": "Câu trích dẫn chính xác từng chữ",
      "page": Số_trang_kiểu_số_nguyên
    }
  ]
}"""

def load_llm():
    global tokenizer, model
    if model is not None:
        return

    with _lock:
        if model is not None:
            return

        import torch  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: PLC0415
        from peft import PeftModel  # noqa: PLC0415

        print("🧠 Đang khởi động AI: Tải Qwen kết hợp Adapter...")
        base_model_id = os.getenv("GENERATOR_MODEL", "Qwen/Qwen2.5-3B-Instruct")
        
        # Đảm bảo bạn đã bỏ thư mục qwen-notebooklm vào models/adapters/
        adapter_path = "models/adapters/qwen-notebooklm"

        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        
        # Nén 4-bit để chạy nhẹ nhàng trên máy bạn
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Gắn Adapter (Kinh nghiệm Fine-tune) vào mô hình gốc
        model = PeftModel.from_pretrained(base_model, adapter_path)
        print("✅ AI đã sẵn sàng nhận lệnh!")

def generate_json_response(user_prompt: str) -> str:
    """Gửi Prompt cho AI và nhận về chuỗi JSON"""
    if model is None:
        load_llm()
        
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    
    # Chuyển đổi hội thoại thành định dạng AI hiểu được
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # Sinh câu trả lời (Nhiệt độ thấp để AI bớt "sáng tạo" và tuân thủ tài liệu)
    outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.1)
    
    # Cắt bỏ phần prompt, chỉ lấy câu trả lời
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()