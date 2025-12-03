# Projeto-Rag com Python, LangChain e Ollama local

O projeto implementa um agente de IA baseado em RAG(Retrieval-Augmented Generation) utilizando Python e langchain, com um uso híbrido capaz de operar tanto com OpenAI API quanto ambiente local via Ollama.

# Recursos
Arquitetura RAG:
  - Leitura de documentos;
  - Geração de embeddings;
  - Vetorização;
  - Recuperação de contexto;
  - Resposta final utilizando LLM.

Fallback automático:
  - Tenta usar OpenAI API caso indisponível, usa o servidor Ollama local.

Aceleração por GPU:
  - Identifica se existe GPU compatível e executa embeddings na GPU sempre que possível, caso não esteja disponivel utiliza CPU.

#Como executar
1. Instale as dependências
  - pip install -r requirements.txt

2. Configurar variáveis opcionais.
  - OPENAI_API_KEY= (Colocar a chave da OPENAI)
  - ollama_host = "(Colocar ip do Ollama local)"
  - model = '(Colocar modelo da IA)'

3. Execute o projeto
  - python main.py
