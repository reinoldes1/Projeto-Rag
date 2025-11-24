from langchain_community.document_loaders import PyPDFDirectoryLoader

BASE_FOLDER = "base"

def create_db():
    documents = load_documents() # Load the documents of the database | Dá load nos documentos da base de dados
    print(documents)
    #chunks = split_chunks(documentos) #Split the documents in chunks | Divide os documentos em chunks de texto
    #vector_chunks(chunks) #Transform the chunks in vectors with embedding | Transforma os chunks em vetores com embedding

def load_documents():
    loader = PyPDFDirectoryLoader(BASE_FOLDER)
    documents = loader.load()
    return documents

def chunks():
    pass

def vector_chunks():
    pass

create_db()