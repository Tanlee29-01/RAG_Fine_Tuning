import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import time
import re
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import fitz  # PyMuPDF

from src.ingestion.pdf_loader import load_pdfs

# Load biến môi trường
load_dotenv()

GITHUB_MODELS_ENDPOINT = "https://models.inference.ai.azure.com"
MODEL_NAME = os.getenv("GITHUB_MODEL", "gpt-4o")


def create_github_models_client() -> OpenAI:
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        raise ValueError("Thiếu biến môi trường GITHUB_TOKEN.")

    return OpenAI(api_key=github_token, base_url=GITHUB_MODELS_ENDPOINT)


client = create_github_models_client()

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

def don_rac_text_slide(text: str) -> str:
    """Hàm chuyên dụng để dọn dẹp các header/footer lặp lại trong Slide"""
    # Xóa các cụm từ rác lặp lại ở mọi trang (Bạn có thể thêm bớt tùy slide)
    tu_khoa_rac = [
        "UNIVERSITY", "UTH OF TRANSPORT", "HOCHIMINH CITY", 
        "TRƯỜNG ĐẠI HỌC GIAO THÔNG VẬN TẢI TP. HỒ CHÍ MINH",
        "[Image", "UTH", "DE TRANSPORT"
    ]
    for tu in tu_khoa_rac:
        # Regex xóa không phân biệt hoa thường
        text = re.sub(re.escape(tu), '', text, flags=re.IGNORECASE)
    
    # Xóa nhiều dấu xuống dòng hoặc khoảng trắng thừa
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text).strip()
    return text

def doc_slide_pdf(file_path: Path):
    """Đọc mỗi slide thành 1 chunk, không cắt nhỏ thêm"""
    doc = fitz.open(file_path)
    pages_data = []
    
    # Với slide, ta thường bắt đầu từ trang 2 (Bỏ qua trang bìa)
    for page_num in range(1, len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        
        # Dọn rác
        text_sach = don_rac_text_slide(text)
        
        # Slide đôi khi chỉ có hình, văn bản rất ngắn -> Bỏ qua
        if len(text_sach) > 50: 
            pages_data.append({"page": page_num + 1, "text": text_sach})
            
    return pages_data

def build_sft_dataset() -> None:
    print("🚀 Bắt đầu tạo dữ liệu tổng hợp cho SLIDE BÀI GIẢNG...")
    
    input_dir = Path("data/raw")
    output_file = Path("data/datasets/sft_train.jsonl")
    pdf_files = load_pdfs(str(input_dir))
    
    if not pdf_files:
        print("⚠️ Không tìm thấy file PDF nào.")
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'a', encoding='utf-8') as f_out:
        for pdf_file in pdf_files:
            print(f"\n🖥️ Đang xử lý Slide: {pdf_file.name}")
            doc_id = pdf_file.stem
            
            # Đọc slide (Mỗi slide là 1 cụm thông tin)
            pages = doc_slide_pdf(pdf_file)
            
            for page_data in pages:
                print(f"  -> Nhờ AI tạo dữ liệu cho Slide số {page_data['page']}...")
                
                # PROMPT ĐẶC BIỆT CHO SLIDE
                prompt_cho_ai = f"""
                Đọc nội dung Slide bài giảng sau (Slide {page_data['page']}, Bài: {doc_id}):
                "{page_data['text']}"
                
                LƯU Ý: Đây là các gạch đầu dòng và từ khóa tóm tắt. Hãy sử dụng tư duy logic để suy luận và liên kết các ý lại với nhau thành câu hoàn chỉnh.
                
                Hãy tạo ra 1 đến 2 cặp Câu hỏi và Câu trả lời học thuật dựa vào nội dung slide này.
                Bỏ qua các thông tin rác như số trang, tên tác giả, tên trường.
                Trả về BẮT BUỘC theo định dạng mảng JSON sau (KHÔNG có markdown):
                [
                  {{
                    "question": "Nội dung câu hỏi?",
                    "answer": "Câu trả lời chi tiết và mạch lạc.",
                    "citation": "Trích dẫn 1 câu ngắn gọn nguyên văn từ slide.",
                    "page": {page_data['page']}
                  }}
                ]
                """
                
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = client.chat.completions.create(
                            model=MODEL_NAME,
                            messages=[{"role": "user", "content": prompt_cho_ai}],
                            temperature=0.4 # Tăng nhẹ nhiệt độ để AI sáng tạo hơn khi nối từ khóa
                        )
                        
                        ket_qua_text = response.choices[0].message.content.strip()
                        if ket_qua_text.startswith("```json"):
                            ket_qua_text = ket_qua_text[7:-3].strip()
                        elif ket_qua_text.startswith("```"):
                            ket_qua_text = ket_qua_text[3:-3].strip()
                            
                        danh_sach_qa = json.loads(ket_qua_text)
                        
                        for qa in danh_sach_qa:
                            user_content = f"NGỮ CẢNH:\n[Trang {qa['page']}] {page_data['text']}\n\nCÂU HỎI:\n{qa['question']}"
                            
                            assistant_response = {
                                "answer": qa["answer"],
                                "citations": [
                                    {
                                        "doc_id": doc_id,
                                        "chunk_id": qa["citation"],
                                        "page": qa["page"],
                                        "score": 1.0
                                    }
                                ]
                            }
                            
                            conversation = {
                                "messages": [
                                    {"role": "system", "content": SYSTEM_PROMPT},
                                    {"role": "user", "content": user_content},
                                    {"role": "assistant", "content": json.dumps(assistant_response, ensure_ascii=False)}
                                ]
                            }
                            
                            f_out.write(json.dumps(conversation, ensure_ascii=False) + '\n')
                        
                        time.sleep(1)
                        break 
                        
                    except Exception as e:
                        print(f"  ❌ Lỗi lần {attempt + 1} ở Slide {page_data['page']}: {e}")
                        time.sleep(3)

    print(f"\n✅ Đã hoàn thành! File dữ liệu lưu tại: {output_file}")


if __name__ == "__main__":
    build_sft_dataset()