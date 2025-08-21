from typing import List
from langchain.schema import Document
from together import Together

class ContextGenerator:
    def __init__(self, together_api_key: str):
        self.client = Together(api_key=together_api_key)
    
    def add_context_to_chunks(self, chunks: List[Document], document_text: str) -> List[Document]:
        contextualized_chunks = []
        
        for chunk in chunks:
            context_prompt = f"""
            Provide 2-3 sentences of context for this chunk from the document:
            
            Document excerpt: {document_text[:1000]}...
            Chunk: {chunk.page_content}
            
            Context:
            """
            
            response = self.client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                messages=[{"role": "user", "content": context_prompt}],
                temperature=0
            )
            context = response.choices[0].message.content
            contextualized_content = f"{context}\n\n{chunk.page_content}"
            contextualized_chunks.append(
                Document(page_content=contextualized_content, metadata=chunk.metadata)
            )
        
        return contextualized_chunks