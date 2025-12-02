from langchain_chroma.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from embeddings import get_embedding
import ollama


load_dotenv()

DB_PATH = "database"
ollama_host = "http://192.168.0.102:30068"
client = ollama.Client(host=ollama_host)
model = 'Qwen3:4b'


prompt_template = """ 
Responda a pergunta do usuário:
{question}
Com base nas seguintes informações:
{context}
Se você não souber a resposta, responda com "Desculpe, não sei a resposta para isso."""

def questions():
    question = input ("Escreva sua pergunta: ")

    # Load database
    embedding = get_embedding()
    db = Chroma(persist_directory=DB_PATH, embedding_function=embedding)

    # Comparing user question (embedding) with database
    results = db.similarity_search_with_relevance_scores(question, k=4)
    if len(results) == 0 or results [0][1] < 0.2:
        print("Could not find the answer in the database")
        return 
    
    answer_results = []
    for result in results:
        text = result[0].page_content
        answer_results.append(text)

    context = "\n\n----\n\n".join(answer_results)

    prompt = ChatPromptTemplate.from_template(prompt_template)
    prompt = prompt.invoke({"question": question, "context": context})
    prompt_str = str(prompt)

    try:
        print("\n----------------------------------------------------------------")
        print(" Utilizando OpenAI para resposta")
        print("----------------------------------------------------------------\n")
        openaimodel = ChatOpenAI()
        answer = openaimodel.invoke(prompt).content
        print("Resposta da IA: ", answer)
    except Exception as e:
        print("\n----------------------------------------------------------------")
        print("OpenAI falhou: ", e)
        print("----------------------------------------------------------------\n")

    try:
        print("\n----------------------------------------------------------------")
        print("Utilizando Ollama local para resposta")
        print("----------------------------------------------------------------\n")
        response = client.generate(model = model, prompt = prompt_str).response
        print("Resposta da IA: ", response)
    except Exception as e:
        print("\n----------------------------------------------------------------")
        print("Ollama falhou: ", e)
        print("----------------------------------------------------------------")

if __name__ == "__main__":
    questions()