from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from openai import OpenAI


client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

loader = DirectoryLoader(path="./", glob="./*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
    add_start_index=True,
)

chunks = text_splitter.split_documents(docs)

model_name = "dangvantuan/sentence-camembert-large"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}

embedding = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

print('model chargé')

qdrant_url = "http://localhost:6333"
collection_name = "test_365"

qdrant = Qdrant.from_documents (
    chunks,
    embedding,
    url = qdrant_url,
    prefer_grpc = False,
    collection_name = collection_name
)

print('db créée')






















