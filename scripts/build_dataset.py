import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ingestion.pdf_loader import load_pdfs  # noqa: E402


load_dotenv()

GITHUB_MODELS_ENDPOINT = "https://models.inference.ai.azure.com"
MODEL_NAME = os.getenv("GITHUB_MODEL", "openai/gpt-4.1-mini")

SYSTEM_PROMPT_GENERATE_QA = """Bạn là một trợ lý AI chuyên tạo dữ liệu huấn luyện từ tài liệu.
Nhiệm vụ của bạn là đọc một đoạn văn bản và sinh ra các cặp câu hỏi - câu trả lời bám sát nội dung.
Quy tắc bắt buộc:
1. Không bịa thêm thông tin ngoài đoạn văn được cung cấp.
2. Câu hỏi phải trả lời được hoàn toàn từ đoạn văn.
3. Câu trả lời phải rõ ràng, ngắn gọn nhưng đủ ý.
4. citation phải là đoạn trích nguyên văn từ chính đoạn văn.
5. Chỉ trả về JSON hợp lệ, không có markdown code block, không có giải thích thêm.
"""

SYSTEM_PROMPT_CHATML = """Bạn là một trợ lý AI phân tích tài liệu.
Nhiệm vụ của bạn là trả lời câu hỏi dựa TRÊN NGỮ CẢNH được cung cấp.
BẮT BUỘC phải trả về kết quả dưới dạng JSON theo đúng cấu trúc sau:
{
  "answer": "Câu trả lời chi tiết của bạn",
  "citations": [
    {
      "doc_id": "Tên_file_hoặc_ID_tài_liệu",
      "chunk_id": "ID của chunk",
      "quote": "Câu trích dẫn chính xác từng chữ",
      "page": 1,
      "score": 1.0
    }
  ]
}"""


def create_github_models_client() -> OpenAI:
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        raise ValueError("Thiếu biến môi trường GITHUB_TOKEN trong file .env")
    return OpenAI(api_key=github_token, base_url=GITHUB_MODELS_ENDPOINT)


client = create_github_models_client()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build synthetic SFT dataset from PDFs")

    parser.add_argument("--input-dir", type=str, default="data/raw", help="Thư mục chứa PDF")
    parser.add_argument(
        "--output-file",
        type=str,
        default="data/datasets/sft_train.jsonl",
        help="File JSONL đầu ra",
    )
    parser.add_argument(
        "--error-file",
        type=str,
        default="data/artifacts/build_dataset_errors.jsonl",
        help="File log lỗi JSONL",
    )

    parser.add_argument("--start-page", type=int, default=10, help="Bắt đầu đọc từ trang thứ mấy, 0-index nội bộ")
    parser.add_argument("--end-page-offset", type=int, default=2, help="Bỏ qua N trang cuối")
    parser.add_argument("--chunk-size", type=int, default=800, help="Kích thước chunk")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="Overlap giữa các chunk")
    parser.add_argument("--min-chunk-length", type=int, default=200, help="Bỏ qua chunk quá ngắn")

    parser.add_argument("--max-samples", type=int, default=50, help="Số sample tối đa muốn sinh")
    parser.add_argument("--max-pages", type=int, default=5, help="Số trang tối đa mỗi PDF sau khi lọc")
    parser.add_argument("--max-chunks-per-page", type=int, default=3, help="Số chunk tối đa mỗi trang")

    parser.add_argument("--qas-per-chunk", type=int, default=2, help="Số QA muốn sinh cho mỗi chunk")
    parser.add_argument("--temperature", type=float, default=0.3, help="Temperature gọi model")
    parser.add_argument("--max-retries", type=int, default=3, help="Số lần retry mỗi chunk")
    parser.add_argument("--sleep-success", type=float, default=1.0, help="Nghỉ sau mỗi request thành công")
    parser.add_argument("--sleep-retry", type=float, default=5.0, help="Nghỉ trước khi retry")

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ghi đè file output thay vì append",
    )

    return parser.parse_args()


def clean_model_json(text: str) -> str:
    text = text.strip()

    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    return text.strip()


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def doc_pdf_bang_pymupdf(
    file_path: Path,
    start_page: int = 10,
    end_page_offset: int = 2,
) -> list[dict[str, Any]]:
    doc = fitz.open(file_path)
    pages_data: list[dict[str, Any]] = []

    total_pages = len(doc)
    if total_pages == 0:
        return []

    if start_page >= total_pages:
        return []

    end_page = total_pages - end_page_offset if total_pages > end_page_offset else total_pages
    end_page = max(start_page, end_page)

    for page_num in range(start_page, end_page):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        text = normalize_whitespace(text)

        if len(text) > 100:
            pages_data.append(
                {
                    "page": page_num + 1,
                    "text": text,
                }
            )

    return pages_data


def build_generation_prompt(doc_id: str, page: int, chunk: str, qas_per_chunk: int) -> str:
    return f"""
Đọc đoạn văn bản sau từ tài liệu "{doc_id}" ở trang {page}:

\"\"\"{chunk}\"\"\"

Hãy tạo CHÍNH XÁC {qas_per_chunk} cặp câu hỏi - câu trả lời dựa hoàn toàn vào đoạn văn trên.

Yêu cầu:
1. Câu hỏi phải trả lời được chỉ từ đoạn văn.
2. Không được suy diễn hay thêm kiến thức ngoài đoạn văn.
3. Câu trả lời phải chính xác, đủ ý.
4. "citation" phải là một đoạn trích nguyên văn từ chính đoạn văn.
5. "page" phải đúng là {page}.
6. Trả về JSON hợp lệ, KHÔNG markdown, theo đúng format:

[
  {{
    "question": "Câu hỏi 1",
    "answer": "Câu trả lời 1",
    "citation": "Đoạn trích nguyên văn",
    "page": {page}
  }}
]
""".strip()


def validate_qa_item(item: dict[str, Any], expected_page: int) -> tuple[bool, str]:
    required_fields = ["question", "answer", "citation", "page"]
    for field in required_fields:
        if field not in item:
            return False, f"Thiếu field '{field}'"

    if not isinstance(item["question"], str) or not item["question"].strip():
        return False, "Field 'question' không hợp lệ"

    if not isinstance(item["answer"], str) or not item["answer"].strip():
        return False, "Field 'answer' không hợp lệ"

    if not isinstance(item["citation"], str) or not item["citation"].strip():
        return False, "Field 'citation' không hợp lệ"

    if safe_int(item["page"], -1) != expected_page:
        return False, f"Field 'page' không khớp expected_page={expected_page}"

    return True, ""


def parse_and_validate_model_output(
    raw_text: str,
    expected_page: int,
    expected_qas: int,
) -> list[dict[str, Any]]:
    cleaned = clean_model_json(raw_text)
    data = json.loads(cleaned)

    if not isinstance(data, list):
        raise ValueError("Kết quả model không phải list JSON")

    if len(data) != expected_qas:
        raise ValueError(f"Model trả {len(data)} QA, expected {expected_qas}")

    validated: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Mỗi phần tử trong list phải là object JSON")

        ok, reason = validate_qa_item(item, expected_page)
        if not ok:
            raise ValueError(reason)

        validated.append(
            {
                "question": normalize_whitespace(item["question"]),
                "answer": normalize_whitespace(item["answer"]),
                "citation": normalize_whitespace(item["citation"]),
                "page": safe_int(item["page"], expected_page),
            }
        )

    return validated


def write_error_record(
    error_file: Path,
    record: dict[str, Any],
) -> None:
    error_file.parent.mkdir(parents=True, exist_ok=True)
    with open(error_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_sft_dataset(
    input_dir: Path,
    output_file: Path,
    error_file: Path,
    start_page: int,
    end_page_offset: int,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_length: int,
    max_samples: int,
    max_pages: int | None,
    max_chunks_per_page: int | None,
    qas_per_chunk: int,
    temperature: float,
    max_retries: int,
    sleep_success: float,
    sleep_retry: float,
    overwrite: bool,
) -> None:
    print("🚀 Bắt đầu tạo synthetic SFT dataset...")

    pdf_files = [Path(p) for p in load_pdfs(str(input_dir))]
    if not pdf_files:
        print(f"⚠️ Không tìm thấy file PDF nào trong thư mục: {input_dir}")
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)
    error_file.parent.mkdir(parents=True, exist_ok=True)

    if overwrite and output_file.exists():
        output_file.unlink()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    mode = "a"
    if overwrite:
        mode = "w"

    written_samples = 0
    processed_pdfs = 0
    processed_pages = 0
    processed_chunks = 0
    success_chunks = 0
    failed_chunks = 0

    with open(output_file, mode, encoding="utf-8") as f_out:
        for pdf_file in pdf_files:
            if written_samples >= max_samples:
                break

            processed_pdfs += 1
            print(f"\n📄 Đang xử lý tài liệu: {pdf_file.name}")
            doc_id = pdf_file.stem

            pages = doc_pdf_bang_pymupdf(
                pdf_file,
                start_page=start_page,
                end_page_offset=end_page_offset,
            )

            if not pages:
                print("  ⚠️ Không có trang hợp lệ sau khi lọc.")
                continue

            if max_pages is not None and max_pages > 0:
                pages = pages[:max_pages]

            for page_data in pages:
                if written_samples >= max_samples:
                    break

                processed_pages += 1
                page_num = page_data["page"]
                page_text = page_data["text"]

                chunks = text_splitter.split_text(page_text)

                if max_chunks_per_page is not None and max_chunks_per_page > 0:
                    chunks = chunks[:max_chunks_per_page]

                for i, chunk in enumerate(chunks, start=1):
                    if written_samples >= max_samples:
                        break

                    chunk = normalize_whitespace(chunk)
                    if len(chunk) < min_chunk_length:
                        continue

                    processed_chunks += 1
                    chunk_uid = f"{doc_id}_p{page_num}_c{i}"

                    print(f"  -> Trang {page_num}, chunk {i}/{len(chunks)} | sample hiện tại: {written_samples}/{max_samples}")

                    prompt = build_generation_prompt(
                        doc_id=doc_id,
                        page=page_num,
                        chunk=chunk,
                        qas_per_chunk=qas_per_chunk,
                    )

                    chunk_success = False
                    last_error = None
                    last_raw_text = None

                    for attempt in range(1, max_retries + 1):
                        try:
                            response = client.chat.completions.create(
                                model=MODEL_NAME,
                                messages=[
                                    {"role": "system", "content": SYSTEM_PROMPT_GENERATE_QA},
                                    {"role": "user", "content": prompt},
                                ],
                                temperature=temperature,
                            )

                            raw_text = (response.choices[0].message.content or "").strip()
                            last_raw_text = raw_text

                            qa_list = parse_and_validate_model_output(
                                raw_text=raw_text,
                                expected_page=page_num,
                                expected_qas=qas_per_chunk,
                            )

                            for qa in qa_list:
                                if written_samples >= max_samples:
                                    break

                                user_content = (
                                    f"NGỮ CẢNH:\n[Trang {qa['page']}] {chunk}\n\n"
                                    f"CÂU HỎI:\n{qa['question']}"
                                )

                                assistant_response = {
                                    "answer": qa["answer"],
                                    "citations": [
                                        {
                                            "doc_id": doc_id,
                                            "chunk_id": chunk_uid,
                                            "quote": qa["citation"],
                                            "page": qa["page"],
                                            "score": 1.0,
                                        }
                                    ],
                                }

                                conversation = {
                                    "messages": [
                                        {"role": "system", "content": SYSTEM_PROMPT_CHATML},
                                        {"role": "user", "content": user_content},
                                        {
                                            "role": "assistant",
                                            "content": json.dumps(assistant_response, ensure_ascii=False),
                                        },
                                    ]
                                }

                                f_out.write(json.dumps(conversation, ensure_ascii=False) + "\n")
                                written_samples += 1

                            success_chunks += 1
                            chunk_success = True
                            time.sleep(sleep_success)
                            break

                        except Exception as e:
                            last_error = str(e)
                            print(f"  ❌ Lỗi lần {attempt}/{max_retries} tại trang {page_num}, chunk {i}: {e}")
                            if attempt < max_retries:
                                time.sleep(sleep_retry)

                    if not chunk_success:
                        failed_chunks += 1
                        write_error_record(
                            error_file=error_file,
                            record={
                                "doc_id": doc_id,
                                "pdf_file": str(pdf_file),
                                "page": page_num,
                                "chunk_id": chunk_uid,
                                "chunk_text": chunk,
                                "error": last_error,
                                "raw_model_output": last_raw_text,
                                "timestamp": int(time.time()),
                            },
                        )

    print("\n✅ Hoàn thành build dataset")
    print(f"📁 Output      : {output_file}")
    print(f"🪵 Error log   : {error_file}")
    print(f"📚 PDFs        : {processed_pdfs}")
    print(f"📄 Pages       : {processed_pages}")
    print(f"🧩 Chunks      : {processed_chunks}")
    print(f"✅ OK chunks    : {success_chunks}")
    print(f"❌ Fail chunks  : {failed_chunks}")
    print(f"📝 Samples     : {written_samples}")


def main() -> None:
    args = parse_args()

    max_pages = args.max_pages if args.max_pages > 0 else None
    max_chunks_per_page = args.max_chunks_per_page if args.max_chunks_per_page > 0 else None

    build_sft_dataset(
        input_dir=Path(args.input_dir),
        output_file=Path(args.output_file),
        error_file=Path(args.error_file),
        start_page=args.start_page,
        end_page_offset=args.end_page_offset,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_chunk_length=args.min_chunk_length,
        max_samples=args.max_samples,
        max_pages=max_pages,
        max_chunks_per_page=max_chunks_per_page,
        qas_per_chunk=args.qas_per_chunk,
        temperature=args.temperature,
        max_retries=args.max_retries,
        sleep_success=args.sleep_success,
        sleep_retry=args.sleep_retry,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()