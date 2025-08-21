# VAIA RAG Multi-Tool Agent

An intelligent RAG pipeline with an agentic AI system that automatically detects user intent and routes queries to the appropriate tool:

##  Tools Available

1. **Q&A Tool**: Answers specific questions about the document
2. **Summarizer Tool**: Provides document summaries  
3. **Data Extractor Tool**: Extracts structured data as JSON

##  Quick Start

### Option 1: Docker Deployment (Recommended)

1. **Set up environment variables:**
   Create a `.env` file with:
   ```
   TOGETHER_API_KEY=your_together_key
   COHERE_API_KEY=your_cohere_key
   ```

2. **Run with Docker Compose:**
   ```bash
   docker-compose up --build
   ```

3. **Access the applications:**
   - **Streamlit UI**: [http://localhost:8501](http://localhost:8501)
   - **FastAPI Backend**: [http://localhost:8000](http://localhost:8000)
   - **API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)

### Option 2: Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   Create a `.env` file with:
   ```
   TOGETHER_API_KEY=your_together_key
   COHERE_API_KEY=your_cohere_key
   ```

3. **Run Streamlit demo:**
   ```bash
   python -m streamlit run src/ui/app.py
   ```

4. **Or run FastAPI backend:**
   ```bash
   python -m uvicorn src.api.main:app --reload
   ```

##  Example Queries

The agent automatically detects intent and routes to the right tool:

**Questions (→ Q&A Tool):**
- "What is FutureFlow's market share?"
- "Who are the main competitors?"
- "How is the company performing?"

**Summaries (→ Summarizer Tool):**
- "Summarize the document"
- "What are the key points?"
- "Give me an overview"

**Data Extraction (→ Extractor Tool):**
- "Extract key financial data"
- "Get market statistics" 
- "Show me the metrics"

## 🧪 Testing the API

Test the FastAPI backend using curl:

```bash
# Q&A Example
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is FutureFlow market share?"}'

# Summarization Example  
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarize the document"}'

# Data Extraction Example
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Extract financial data"}'
```

## 📁 Project Structure

```
vaia-rag/
├── src/
│   ├── core/
│   │   ├── document_loader.py    # Document loading & chunking
│   │   ├── context_generator.py  # Contextual chunking
│   │   ├── vector_store.py       # FAISS embeddings & search
│   │   ├── reranker.py          # Cohere reranking
│   │   └── rag_pipeline.py      # Main RAG orchestrator
│   ├── agents/
│   │   ├── tools.py             # LangChain tools (@tool decorators)
│   │   └── agent.py             # Multi-tool agent with LangGraph
│   ├── api/
│   │   └── main.py              # FastAPI backend
│   └── ui/
│       └── app.py               # Streamlit frontend
├── data/
│   └── document.txt             # Sample document
├── Dockerfile                   # Container configuration
├── docker-compose.yml          # Multi-service deployment
└── requirements.txt             # Python dependencies
```

## 🎯 Key Features

- **Intelligent routing**: Agent automatically detects user intent and selects the right tool.
- **Tool-specific prompts**: Optimized prompts per tool. Few-shot prompting for the Data Extractor enforces strict JSON output.
- **Contextual chunking**: Adds additional context to each chunk for better retrieval than naive size-based chunking.
- **Reranking**: Uses Cohere to rank top-k (k=5) results for better answers.
- **RAG pipeline**: Separate modules for each stage to keep the system scalable.

```
📄 Document → ✂️ Chunks → 🧠 Context → 🔢 Vectors → 💾 Store
                                                        ↓
👤 User Query → 🔍 Search → 📊 Rerank → 🤖 LLM → ✅ Answer
```

### 1. Document ingestion

```
"Innovate Inc. has 12% market share"
    ↓ (chunk)
"This chunk discusses market share data for Innovate Inc. Content: Innovate Inc. has 12% market share"
    ↓ (embed)
[0.1, 0.4, 0.7, ...] (vector representation)
```

### 2. Retrieving relevant results

```
User: "What is market share?"
    ↓ (embed query)
[0.2, 0.3, 0.8, ...]
    ↓ (similarity search)
Find chunks with similar vectors
    ↓ (rerank)
Return top 5 most relevant chunks
```

