from typing import Dict, Any
import json
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from src.core.rag_pipeline import RAGPipeline

# Global RAG pipeline instance
rag_pipeline = None
llm_client = None

def initialize_rag_pipeline(together_key: str, cohere_key: str, document_path: str, llm=None):
    """Initialize the RAG pipeline and share LLM client"""
    global rag_pipeline, llm_client
    rag_pipeline = RAGPipeline(together_key, cohere_key)
    rag_pipeline.process_document(document_path)
    llm_client = llm


@tool
def qa_tool(question: str) -> str:
    """Answer specific questions about the document content"""
    if not rag_pipeline or not llm_client:
        return "RAG pipeline not initialized"
    
    docs = rag_pipeline.get_relevant_docs(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    prompt = f"""You are a Q&A specialist. Answer the question using only the provided context.

Context:
{context}

Question: {question}

Answer:"""

    response = llm_client.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


@tool  
def summarizer_tool(query: str) -> str:
    """Create summaries of the document"""
    if not rag_pipeline or not llm_client:
        return "RAG pipeline not initialized"
    
    docs = rag_pipeline.get_relevant_docs("document overview summary", k=5)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    prompt = f"""Summarize the document based on the following request.

Context:
{context}

User Request: {query}

Summary:"""

    response = llm_client.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


@tool
def data_extractor_tool(extraction_query: str) -> str:
    """Extract data points in JSON format based on user query"""
    if not rag_pipeline or not llm_client:
        return '{"error": "RAG pipeline not initialized"}'
    
    docs = rag_pipeline.get_relevant_docs(extraction_query, k=5)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    prompt = f"""Extract entities and their values from the context. Return as JSON with entity:value pairs.

Examples:

Context: "The company's revenue grew by 25% to $50 million in 2023."
Query: "Extract financial data"
JSON: {{"revenue": "$50 million", "growth_rate": "25%", "year": "2023"}}

Context: "Apple Inc. was founded by Steve Jobs. Market cap is $2.8 trillion."
Query: "Extract company information"  
JSON: {{"company_name": "Apple Inc.", "founder": "Steve Jobs", "market_cap": "$2.8 trillion"}}

Context: "The meeting is scheduled for 3 PM on March 15th in Conference Room A."
Query: "Extract meeting details"
JSON: {{"time": "3 PM", "date": "March 15th", "location": "Conference Room A"}}

Instructions:
1. Identify entities (people, companies, amounts, dates, percentages, etc.)
2. Extract their values from the context
3. Return as entity:value pairs in JSON format
4. Use descriptive entity names
5. Return ONLY valid JSON

Context: {context}
Query: {extraction_query}

JSON:"""

    response = llm_client.invoke([HumanMessage(content=prompt)])
    result = response.content.strip()
    
    try:
        parsed = json.loads(result)
        return json.dumps(parsed)  
    except:
        return json.dumps({"extracted_info": result})