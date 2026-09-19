import os
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# chạy không thành công: bị mắc lỗi ValueError: Batch size of 5541 is greater than max batch size of 5461



# ================= ================= =================
# 1. CẤU HÌNH & KHỞI TẠO CLIENT
# ================= ================= =================
LM_STUDIO_URL = "http://localhost:1234/v1"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHAT_MODEL = "qwen2.5-1.5b-instruct"
DATA_FOLDER = "./documents"          # Thư mục chứa tài liệu cục bộ
CHROMA_PATH = "./chroma-db"     # Thư mục lưu cơ sở dữ liệu vector

client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")


# Class tùy chỉnh để ChromaDB gọi Embedding API từ LM Studio
# class LMStudioEmbeddingFunction:
#     def __init__(self, model_name=EMBEDDING_MODEL):
#         self.model_name = model_name

#     def __call__(self, input):
#         if isinstance(input, str):
#             input = [input]
#         response = client.embeddings.create(
#             model=self.model_name,
#             input=input
#         )
#         return [data.embedding for data in response.data]

#     def name(self) -> str:
#         return f"lm_studio_{self.model_name}"

# ================= ================= =================
# 2. XỬ LÝ & ĐỌC DỮ LIỆU CỤC BỘ (DOCUMENTS LOADING)
# ================= ================= =================
def load_documents_from_folder(folder_path):
    """Đọc toàn bộ file .txt, .pdf, .md trong thư mục target"""
    documents = []
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"📁 Đã tạo thư mục '{folder_path}'. Hãy thả các file .txt, .pdf hoặc .md vào đây!")
        return documents

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        text = ""
        
        # Đọc file TXT và Markdown
        if file_name.endswith(".txt") or file_name.endswith(".md"):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                
        # Đọc file PDF
        elif file_name.endswith(".pdf"):
            reader = PdfReader(file_path)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
                    
        if text.strip():
            documents.append({"source": file_name, "content": text})
            print(f"📄 Đã đọc: {file_name}")

    return documents


# ================= ================= =================
# 3. CHUNKING & NẠP VÀO CHROMADB
# ================= ================= =================
def setup_vector_db():
    """Đọc tài liệu, chia đoạn (chunking) và lưu vào ChromaDB"""
    # Khởi tạo ChromaDB Persistent Client
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    # embedding_fn = LMStudioEmbeddingFunction()
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    collection = chroma_client.get_or_create_collection(
        name="local_documents",
        embedding_function=embedding_fn
    )

    # Nếu collection đã có dữ liệu thì không nạp lại
    if collection.count() > 0:
        print(f"✅ Đã kết nối ChromaDB (Đang có {collection.count()} chunks dữ liệu).")
        return collection

    # Đọc dữ liệu từ thư mục
    raw_docs = load_documents_from_folder(DATA_FOLDER)
    if not raw_docs:
        print("⚠️ Chưa có tài liệu nào trong thư mục data.")
        return collection

    # Cấu hình Chunking (Mỗi đoạn ~500 ký tự, đè phủ 50 ký tự)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = []
    metadatas = []
    ids = []
    chunk_counter = 0

    for doc in raw_docs:
        sub_chunks = text_splitter.split_text(doc["content"])
        for chunk in sub_chunks:
            chunks.append(chunk)
            metadatas.append({"source": doc["source"]})
            ids.append(f"chunk_{chunk_counter}")
            chunk_counter += 1

    # Đẩy dữ liệu vào ChromaDB
    print(f"⏳ Đang nhúng {len(chunks)} chunks vào ChromaDB...")
    collection.add(
        documents=chunks,
        metadatas=metadatas,
        ids=ids
    )
    print("✅ Đã hoàn tất đưa dữ liệu vào ChromaDB!")
    return collection


# ================= ================= =================
# 4. TRUY VẤN NGUYÊN CẢNH & HỎI ĐÁP (RAG CORE)
# ================= ================= =================
def ask_question(collection, query: str):
    # Tìm 3 đoạn văn bản liên quan nhất (top_k = 3)
    results = collection.query(
        query_texts=[query],
        n_results=3
    )

    retrieved_docs = results["documents"][0] if results["documents"] else []
    retrieved_meta = results["metadatas"][0] if results["metadatas"] else []

    # Ghép bối cảnh (context) kèm nguồn file
    context_text = ""
    for idx, doc in enumerate(retrieved_docs):
        source = retrieved_meta[idx].get("source", "N/A")
        context_text += f"--- Nguồn: {source} ---\n{doc}\n\n"

    # Tạo Prompt chứa Context
    system_prompt = (
        "Bạn là một trợ lý AI phân tích tài liệu thông minh. "
        "Hãy trả lời câu hỏi dựa HOÀN TOÀN vào phần Ngữ cảnh được cung cấp. "
        "Nếu thông tin không có trong ngữ cảnh, hãy trả lời rõ ràng: 'Tôi không tìm thấy thông tin này trong tài liệu cục bộ.'"
    )
    
    user_prompt = f"Ngữ cảnh:\n{context_text}\nCâu hỏi: {query}"

    # Gọi LLM Qwen2.5 trả lời
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content, context_text


# ================= ================= =================
# 5. CHƯƠNG TRÌNH CHÍNH (CLI CHATBOT)
# ================= ================= =================
if __name__ == "__main__":
    print("🚀 Đang khởi động hệ thống Local RAG...")
    collection = setup_vector_db()

    print("\n" + "="*50)
    print("🤖 RAG CHATBOT ĐÃ SẴN SÀNG! (Gõ 'exit' hoặc 'quit' để thoát)")
    print("="*50 + "\n")

    while True:
        user_input = input("\n👤 Bạn hỏi: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Chương trình kết thúc.")
            break
        if not user_input:
            continue

        answer, context = ask_question(collection, user_input)
        
        print("\n🤖 Qwen2.5 Trả lời:")
        print(answer)