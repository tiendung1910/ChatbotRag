import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

LM_STUDIO_URL = "http://localhost:1234/v1"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHAT_MODEL = "qwen2.5-1.5b-instruct"
DATA_FOLDER = "./documents"
VECTORDB_DIR = "./vector-db"


def load_documents(folder_path):
    documents = []
    
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path,file_name)
        text = ""

        if file_name.endswith(".pdf"):
            reader = PdfReader(file_path)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            
        if text.strip():
            documents.append({"source":file_name,"content":text})
            print(f"{file_name} ")
        
    return documents

# chunking vs nap chromadb
def setup_vector_db():
    client = chromadb.PersistentClient(VECTORDB_DIR)
    embedding_ft = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name = "all-MiniLM-L6-v2"
    )
    collection = client.get_or_create_collection("chunks-table")
    
    if(collection.count() > 0):
        print(f"Đã nạp tài liệu rồi. Đang có {collection.count()} chunk")
        return collection
        
    
    raw_docs = load_documents(DATA_FOLDER)
    
    if not raw_docs:
        print("khong tim thay tai lieu")
        return collection
    
    # cấu hình chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = []
    metadatas = []
    ids = []
    chunk_counter = 0
    BATCH_SIZE = 1000
    
    for doc in raw_docs:
        sub_chunks = text_splitter.split_text(doc["content"])
        for chunk in sub_chunks:
            chunks.append(chunk)
            metadatas.append({"source": doc["content"]})
            ids.append(f"chunk_{chunk_counter}")
            chunk_counter += 1

    for i in range(0, len(chunks), BATCH_SIZE):
        batch_chunk = chunks[i : i+BATCH_SIZE]
        batch_metadata = metadatas[i : i+BATCH_SIZE]
        batch_ids = ids[i : i+BATCH_SIZE]
        collection.add(ids=batch_ids,documents=batch_chunk,metadatas=batch_metadata)
    print(f"lưu data chưa đính kèm metadata, thành công")
    
    return collection

def queryAI(collection, query: str):
    results = collection.query(
        query_texts=[query],
        n_results=3
    )
    
    retrieve_docs = results["documents"][0] if results["documents"] else []
    retrieve_metadata = results["metadatas"][0] if results["metadatas"] else []

    

if __name__ == "__main__":
    print("Đang khởi động bot chat")
    collection = setup_vector_db()
    
    print("🤖Bot đã sẵn sàng !!!")
    while True:
        question = input("\nBạn hỏi: ").strip()
        if question.lower() == "quit" or question.lower() == exit:
            print("\nTạm biệt bạn")
            break
        
        if not question:
            continue
        
        print("\nĐang xử lý câu hỏi:")
        queryAI(collection,question)
        