"""Interactive query demo — search the FAISS vector store with a question."""
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

load_dotenv()

print("🔍 Đang khởi động hệ thống tìm kiếm...")
model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
embeddings = HuggingFaceEmbeddings(model_name=model_name)

try:
    vector_db = FAISS.load_local("data/vector_db", embeddings, allow_dangerous_deserialization=True)
except Exception:
    print("❌ Lỗi: Chưa tìm thấy CSDL. Bạn đã chạy 'make index' chưa?")
    exit(1)

while True:
    cau_hoi = input("\n🤔 Nhập câu hỏi (hoặc gõ 'exit' để thoát): ")
    if cau_hoi.lower() == "exit":
        break

    print("Đang tìm tài liệu...")
    ket_qua = vector_db.similarity_search(cau_hoi, k=3)

    print("\n" + "=" * 50)
    for i, doc in enumerate(ket_qua):
        print(f"--- 📍 TRÍCH DẪN {i+1} (Trang {doc.metadata.get('page', 'Không rõ')}) ---")
        print(doc.page_content)
    print("=" * 50)
