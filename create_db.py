from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import torch

load_dotenv()

BASE_FOLDER = "base"

def create_db():
    documents = load_documents() # Load the documents of the database | Dá load nos documentos da base de dados
    chunks = split_chunks(documents) #Split the documents in chunks | Divide os documentos em chunks de texto
    vector_chunks(chunks) #Transform the chunks in vectors with embedding | Transforma os chunks em vetores com embedding

def load_documents():
    loader = PyPDFDirectoryLoader(BASE_FOLDER)
    documents = loader.load()
    return documents

def split_chunks(documents):
    documents_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50,
        length_function = len,
        add_start_index = True
    )
    chunks = documents_splitter.split_documents(documents)
    print(f"{len(chunks)} Chunks created")
    return chunks

def vector_chunks(chunks):
    try:
        database = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory = "database")
        print("Banco de dados Criado")
    except Exception as e:
        print("Erro OpenAI embeddings:", e)
        print("Usando fallback local com sentence-transformers")
        try:
            from sentence_transformers import SentenceTransformer

            class HFEmbeddings:
                def __init__(self, model_name="all-MiniLM-L6-v2", device=None, batch_size=32):
                    if device is None:
                        device ="cuda" if torch.cuda.is_available()  else "cpu"
                        if device == "cuda":
                            print("Usando GPU para o Embeddings")
                        else:
                            print("Usando CPU para o Embeddings")
                    self.device = device
                    self.batch_size = batch_size
                    self.model = SentenceTransformer(model_name)

                def embed_documents(self, documents):
                    texts = [d.page_content if hasattr(d, "page_content") else str(d) for d in documents]
                    embs = self.model.encode(texts, show_progress_bar=False)
                    return [list(map(float, e)) for e in embs]

                def embed_query(self, text):
                    v = self.model.encode([text], show_progress_bar=False)[0]
                    return list(map(float, v))

            hf = HFEmbeddings()
            db = Chroma.from_documents(chunks, hf, persist_directory="database")
            print("Banco de dados criado com sentence-transformers local")
        except Exception as e2:
            print("Falha no fallback local:", e2)


create_db()