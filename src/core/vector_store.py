from typing import List
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from together import Together

class TogetherEmbeddings(Embeddings):
    def __init__(self, together_api_key: str):
        self.client = Together(api_key=together_api_key)
        self.model = "BAAI/bge-large-en-v1.5"
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            response = self.client.embeddings.create(model=self.model, input=text)
            embeddings.append(response.data[0].embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        response = self.client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding

class VectorStore:
    def __init__(self, together_api_key: str):
        self.embeddings = TogetherEmbeddings(together_api_key)
    
    def create_vectorstore(self, documents: List[Document]) -> FAISS:
        return FAISS.from_documents(documents, self.embeddings)
    
    def similarity_search(self, vectorstore: FAISS, query: str, k: int = 10) -> List[Document]:
        return vectorstore.similarity_search(query, k=k)
    
