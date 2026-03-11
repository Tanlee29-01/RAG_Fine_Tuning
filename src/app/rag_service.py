import os
import json
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from src.llm.generator import generate_json_response

load_dotenv()

# Biến toàn cục lưu Vector DB
vector_db = None

def load_vector_db():
    global vector_db
    if vector_db is not None:
        return
        
    print("🔍 Đang tải Bộ não Vector (FAISS)...")
    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vector_db = FAISS.load_local("data/vector_db", embeddings, allow_dangerous_deserialization=True)
    print("✅ Vector DB đã sẵn sàng!")

def run_rag(question: str) -> dict:
    """Nhận câu hỏi -> Tìm tài liệu -> Nhờ AI trả lời -> Trả về JSON chuẩn"""
    if vector_db is None:
        load_vector_db()

    # 1. TÌM KIẾM TÀI LIỆU (Retrieval)
    # Lấy 3 đoạn văn bản liên quan nhất
    docs = vector_db.similarity_search(question, k=3)
    
    # Gộp các đoạn văn bản lại thành "Ngữ cảnh"
    context_text = ""
    for i, doc in enumerate(docs):
        page = doc.metadata.get('page', 'Không rõ')
        context_text += f"[Trang {page}] {doc.page_content}\n\n"

    # 2. GHÉP PROMPT (Augmented)
    user_content = f"NGỮ CẢNH:\n{context_text}\nCÂU HỎI:\n{question}"

    # 3. SINH CÂU TRẢ LỜI (Generation)
    print("🤖 Đang nhờ AI suy nghĩ câu trả lời...")
    raw_response = generate_json_response(user_content)

    # 4. XỬ LÝ LỖI JSON (Phòng hờ AI sinh thêm ký tự thừa)
    try:
        if raw_response.startswith("```json"):
            raw_response = raw_response[7:-3].strip()
        elif raw_response.startswith("```"):
            raw_response = raw_response[3:-3].strip()
            
        result_json = json.loads(raw_response)
        return result_json
    except json.JSONDecodeError:
        print("⚠️ AI trả về sai định dạng JSON!")
        return {
            "answer": raw_response,
            "citations": []
        }