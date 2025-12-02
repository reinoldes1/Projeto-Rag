from dotenv import load_dotenv
import torch

load_dotenv()

def get_embedding():
    try:
        from langchain_openai import OpenAIEmbeddings
        emb = OpenAIEmbeddings()
        try:
            emb.embeded_query("Teste")
            print("Usando OpenAIEmbeddings")
            return emb
        except Exception as e:
            print("OpenAIEmbeddings instanciado mas falhou no teste: ", e)
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
                embs = self.model.encode(texts, show_progress_bar=True)
                return [list(map(float, e)) for e in embs]

            def embed_query(self, text):
                v = self.model.encode([text], show_progress_bar=True)[0]
                return list(map(float, v))    
                       
        return HFEmbeddings()
    except Exception as e2:
        print("Falha no fallback local: ", e2)
        raise