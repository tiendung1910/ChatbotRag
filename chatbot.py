import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import os
from pypdf import PdfReader
from langchain_text_splitters import SentenceTransformersTokenTextSplitter

LM_STUDIO_URL = "http://localhost:1234/v1"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHAT_MODEL = "qwen2.5-1.5b-instruct"
DATA_FOLDER = "./documents"
VECTORDB_DIR = "./vector-db"

client = OpenAI(base_url=LM_STUDIO_URL,api_key="lm-studio")


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
    client = chromadb.PersistentClient('./vector-db')
    embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL)
    
    collection = client.get_or_create_collection(
        name="vector-table",
        embedding_function=embedding_model
    )
    
    if collection.count() > 0:
        print(f"dang co {collection.count()} chunk data")
        return collection
    
    raw_document = load_documents(DATA_FOLDER)
    if not raw_document:
        print("chua co tai lieu")
        return collection
    
    text_splitter = SentenceTransformersTokenTextSplitter(
        tokens_per_chunk=100,
        chunk_overlap=20
    )
    
    chunks = []
    metadatas = []
    ids = []
    chunk_counter = 0
    for doc in raw_document:
        sub_chunks = text_splitter.split_text(doc)
        for chunk in sub_chunks:
            chunks.append(chunk)
            metadatas.append({"source": doc["source"]})
            chunk_counter += 1
            
    print(f"ddang nhung {len(chunk)} vao vector db")
    collection.add(
        documents=chunks,
        metadatas=metadatas,
        ids=ids
    )
    
    
    

if __name__ == "__main__":
    load_documents(DATA_FOLDER)
    setup_vector_db()
    # pass