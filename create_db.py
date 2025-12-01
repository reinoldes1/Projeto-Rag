from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from dotenv import load_dotenv
from embeddings import get_embedding

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
        chunk_size = 2000,
        chunk_overlap = 500,
        length_function = len,
        add_start_index = True
    )
    chunks = documents_splitter.split_documents(documents)
    print(f"{len(chunks)} Chunks created")
    return chunks

def vector_chunks(chunks):
    try:
        embedding = get_embedding()
        database = Chroma.from_documents(chunks, embedding, persist_directory = "database")
        print("Banco de dados Criado")
    except Exception as e:
        print("Erro ao criar banco de dados: ", e)
    


create_db()