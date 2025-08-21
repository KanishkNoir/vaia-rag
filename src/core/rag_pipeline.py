from src.core.document_loader import DocumentLoader
from src.core.context_generator import ContextGenerator
from src.core.vector_store import VectorStore
from src.core.reranker import Reranker

class RAGPipeline:
    def __init__(self, together_api_key: str, cohere_api_key: str):
        self.document_loader = DocumentLoader()
        self.context_generator = ContextGenerator(together_api_key)
        self.vector_store = VectorStore(together_api_key)
        self.reranker = Reranker(cohere_api_key)
        self.vectorstore = None
    
    def process_document(self, document_path: str):
        """Process document and create vectorstore"""
        
        # Load and chunk document
        document_text = self.document_loader.load_text_file(document_path)
        chunks = self.document_loader.create_chunks(document_text)
        
        # Add context to chunks
        contextualized_chunks = self.context_generator.add_context_to_chunks(chunks, document_text)
        
        # Create vectorstore (in-memory, no persistence)
        self.vectorstore = self.vector_store.create_vectorstore(contextualized_chunks)
    
    def get_relevant_docs(self, question: str, k: int = 10):
        """Get relevant documents for a query"""
        if not self.vectorstore:
            raise ValueError("Document not processed yet. Call process_document() first.")
        
        # Retrieve and rerank documents
        similar_docs = self.vector_store.similarity_search(self.vectorstore, question, k=k)
        return self.reranker.rerank_documents(question, similar_docs)
