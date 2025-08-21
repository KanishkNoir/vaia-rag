import cohere
from typing import List
from langchain.schema import Document

class Reranker:
    def __init__(self, cohere_api_key: str):
        self.cohere_client = cohere.Client(cohere_api_key)
    
    def rerank_documents(self, query: str, documents: List[Document], top_k: int = 3) -> List[Document]:
        doc_contents = [doc.page_content for doc in documents]
        
        try:
            reranked = self.cohere_client.rerank(
                model="rerank-english-v2.0",
                query=query,
                documents=doc_contents,
                top_n=top_k
            )
            return [documents[result.index] for result in reranked.results]
        except:
            return documents[:top_k]