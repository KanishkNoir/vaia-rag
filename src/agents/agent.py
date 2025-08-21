from langchain_together import ChatTogether
from src.agents.tools import qa_tool, summarizer_tool, data_extractor_tool, initialize_rag_pipeline

class MultiToolAgent:
    def __init__(self, together_key: str, cohere_key: str, document_path: str):
        # Create LLM for intent detection
        self.llm = ChatTogether(
            together_api_key=together_key,
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            temperature=0
        )
        
        # Initialize RAG pipeline and share LLM instance
        initialize_rag_pipeline(together_key, cohere_key, document_path, self.llm)
    
    def query(self, question: str):
        """Intent-based routing - return tool results directly"""
        try:
            # Use LLM to detect user intent
            intent = self._detect_intent(question)
            
            # Route based on intent
            if intent == "data_extraction":
                result = data_extractor_tool.invoke({"extraction_query": question})
                
            elif intent == "summarization":
                result = summarizer_tool.invoke({"query": question})
                
            else:  # qa
                result = qa_tool.invoke({"question": question})
            
            return result
            
        except Exception as e:
            return f"Error processing query: {str(e)}"
    
    def _detect_intent(self, question: str) -> str:
        """Detect user intent using LLM"""
        from langchain_core.messages import HumanMessage
        
        intent_prompt = f"""Classify the user's intent. Respond with exactly one word:

- "data_extraction" if user wants to extract structured data, find specific information, get statistics, or retrieve data points
- "summarization" if user wants a summary, overview, or condensed version of content  
- "qa" if user is asking a specific question about the document

Examples:
"Extract financial data" → data_extraction
"Get market statistics" → data_extraction
"Find company information" → data_extraction
"Summarize the document" → summarization
"Give me an overview" → summarization
"What is the market share?" → qa
"Who are the competitors?" → qa

User query: "{question}"

Intent:"""

        response = self.llm.invoke([HumanMessage(content=intent_prompt)])
        intent = response.content.strip().lower()
        
        # Ensure valid intent
        if intent not in ["data_extraction", "summarization", "qa"]:
            intent = "qa"  # Default fallback
            
        return intent
