from functools import lru_cache
from pinecone import Pinecone, ServerlessSpec
from medical_chatbot.config import settings
from langchain_huggingface import HuggingFaceEmbeddings

#Initialize Pinecone and embedding clients
@lru_cache(maxsize=1)
def clients() -> tuple[Pinecone, HuggingFaceEmbeddings]:
    if not settings.PINECONE_API_KEY :
        raise ValueError("Pinecone API key must be set in the environment variables.")
    
    pc = Pinecone(api_key=settings.PINECONE_API_KEY.get_secret_value())
    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
    
    return pc, embeddings

#caching the embedding dimension to avoid repeated computation
@lru_cache(maxsize=1)
def embedding_dimension() -> int:
    _, embeddings = clients()
    return len(embeddings.embed_query("dimension check"))

#Create the Pinecone index if it doesn't exist
def ensure_index():
    pc, embeddings = clients()
    index_name = settings.PINECONE_INDEX_NAME
    dimension = embedding_dimension()
    
    if not pc.has_index(index_name):
        pc.create_index(name=index_name,
                        dimension=dimension,
                        metric="cosine",
                        spec=ServerlessSpec(cloud='aws', region='us-east-1'))
    return pc.Index(index_name)    

#Convert text into numerical vectors
def embed(texts):
    _, embeddings =  clients()
    return embeddings.embed_documents(texts)

#Search the knowledge base for relevant information based on the query
def search_kb(query, top_k=5):
    index = ensure_index()
    _, embeddings = clients()
    vector = embeddings.embed_query(query)
    result = index.query(vector=vector, top_k=top_k, include_metadata=True)
    return [{'page': match.metadata.get('page'),
             'text': match.metadata.get('text',''),
             'score': round(match.score, 3),
             'source': match.metadata.get('source', 'unknown') }
            for match in result.matches if match.score is not None and match.score > 0.20]
   