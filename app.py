from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client import QdrantClient
from openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
# from llama_index import PromptTemplate
# from llama_index import ServiceContext
from langchain.chains import RetrievalQA
from llama_index.llms.llama_cpp import LlamaCPP
# from IPython.display import Markdown

model_name = "dangvantuan/sentence-camembert-large"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}

embedding = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

qdrant_url = "http://localhost:6333"
collection_name = "test_365"

client = QdrantClient(
    url = qdrant_url,
    prefer_grpc = False
)

print(client)
print('---------------------')

db = Qdrant(
    client = client,
    embeddings = embedding,
    collection_name = collection_name
)

print(db)
print('----------------------')

# query = "Qu'est ce qu'un client ?"

# retriever = db.similarity_search_with_score(query=query, k=5)

# for i in retriever:
#     doc, score = i
#     print({"score": score, "content": doc.page_content, "metadata": doc.metadata})

llm = LlamaCPP(
    model_path="./mistral-7b-instruct-v0.1.Q5_0.gguf",
    temperature=0.1,
    max_new_tokens=2048,
    context_window=3900,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": 0},
    verbose=True,
)

print ('LLM chargé')

retriever = db.as_retriever(search_kwargs={"k": 3})

template = """Tu es un expert dans l'utilisation du logiciel Microsoft Dynamics 365. Appui toi sur ce contexte pour répondre à la question de l'utilisateur, si tu ne sais pas répond 'je n'ai pas les information', n'essai pas d'inventer une réponse:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

print('prompt ok')

qa_chain = RetrievalQA.from_chain_type(llm = llm,
                                       chain_type="stuff",
                                       retriever = retriever,
                                       chain_type_kwargs={"prompt": prompt},
                                       return_source_documents = True,
                                       verbose = True
                                       )

qa_chain("Qu'est ce qu'un client ?")











"""qa_chain = RetrievalQA.from_chain_type(llm = "local-model",
                                       retriever = retriever,
                                       chain_type_kwargs={"prompt": prompt},
                                       return_source_documents = True,
                                       verbose = True
                                       )"""