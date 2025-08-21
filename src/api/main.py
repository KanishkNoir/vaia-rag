import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from src.agents.agent import MultiToolAgent

load_dotenv()
app = FastAPI(title="RAG Multi-Tool Agent API")

class QueryRequest(BaseModel):
    query: str

# Initialize agent
together_key = os.getenv("TOGETHER_API_KEY")
cohere_key = os.getenv("COHERE_API_KEY")
agent = MultiToolAgent(together_key, cohere_key, "data/document.txt")

@app.post("/query")
async def query_agent(request: QueryRequest):
    """Query the multi-tool agent"""
    response = agent.query(request.query)
    
    # For data extraction, try to return JSON directly
    try:
        import json
        # If response is a JSON string, parse and return as object
        parsed = json.loads(response)
        return parsed
    except:
        # For non-JSON responses (QA, summaries), return as text
        return {"response": response}

@app.get("/")
async def root():
    return {"message": "RAG Multi-Tool Agent API"}
