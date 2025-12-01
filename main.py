from langchain_chroma.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from embeddings import get_embedding


load_dotenv()

DB_PATH = "database"

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
    results = db.similarity_search_with_relevance_scores(question, k=3)
    if len(results) == 0 or results [0][1] < 0.7:
        print("Could not find the answer in the database")
        return 
    
    answer_results = []
    for result in results:
        text = result.page_content
        answer_results.append(text)

    context = "\n\n----\n\n".join(answer_results)

    prompt = ChatPromptTemplate.from_template(prompt_template)
    prompt = prompt.invoke({"question": question, "context": context})