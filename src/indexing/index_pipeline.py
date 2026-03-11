import os
from pathlib import Path
from dotenv import load_dotenv

# Import các công cụ "phép thuật" từ LangChain
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from src.ingestion.pdf_loader import load_pdfs

# Load cấu hình từ file .env (chứa tên mô hình embedding)
load_dotenv()

def run_index_pipeline() -> None:
    print("🚀 Bắt đầu xây dựng Cơ sở dữ liệu Vector (Bộ não tìm kiếm)...")
    
    input_dir = Path("data/raw")
    db_dir = Path("data/vector_db") # Thư mục lưu não bộ
    
    # 1. Quét tìm tài liệu PDF
    pdf_files = load_pdfs(str(input_dir))
    if not pdf_files:
        print(f"⚠️ Không tìm thấy tài liệu nào trong {input_dir}.")
        return

    # 2. Đọc nội dung PDF
    documents = []
    for pdf in pdf_files:
        print(f"📄 Đang đọc file: {pdf.name}")
        # Dùng PyMuPDFLoader của LangChain cho nhanh gọn và giữ được số trang
        loader = PyMuPDFLoader(str(pdf))
        docs = loader.load()
        documents.extend(docs)

    # 3. Cắt nhỏ văn bản (Chunking)
    print(f"✂️ Đang chia nhỏ {len(documents)} trang tài liệu...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Đã tạo được {len(chunks)} đoạn văn bản (chunks).")

    # 4. Tải mô hình biến chữ thành số (Embedding Model)
    # Theo file .env của bạn, nó sẽ tải model: sentence-transformers/all-MiniLM-L6-v2
    embedding_model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    print(f"🧠 Đang tải mô hình ngôn ngữ Embedding: {embedding_model_name}")
    print("   (Có thể mất chút thời gian tải model lần đầu...)")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    # 5. Lưu vào Cơ sở dữ liệu FAISS
    print("💾 Đang chuyển đổi văn bản thành Vector và lưu vào CSDL FAISS...")
    vector_db = FAISS.from_documents(chunks, embeddings)

    # Tạo thư mục và lưu xuống ổ cứng
    db_dir.mkdir(parents=True, exist_ok=True)
    vector_db.save_local(str(db_dir))

    print(f"🎉 Hoàn tất! Bộ não Vector đã được lưu tại: {db_dir}")

    