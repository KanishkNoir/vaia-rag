# VAIA RAG Multi-Tool Agent

##  Tools Available

1. **Q&A Tool**: Answers specific questions about the document
2. **Summarizer Tool**: Provides document summaries  
3. **Data Extractor Tool**: Extracts structured data as JSON

##  Deployment

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


##  Key Features

- **Intelligent routing**: Agent automatically detects user intent and selects the right tool.
- **Tool-specific prompts**: Optimized prompts per tool. Few-shot prompting for the Data Extractor enforces strict JSON output.
- **Contextual chunking**: Adds additional context to each chunk for better retrieval than naive size-based chunking.
- **Reranking**: Uses Cohere to rank top-k (k=5) results for better answers.
- **RAG pipeline**: Separate modules for each stage to keep the system scalable.

```
 Document →  Chunks →  Context →  Vectors →  Store
                                                        ↓
 User Query →  Search →  Rerank →  LLM →  Answer
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

## Design Decision

### 1. Instead of implementing the whole RAG pipeline in a single file I seperated each step in its own file due to scalability and cleaner code.

### 2. Why Together AI:
Due to pricing issue of openai api I used Together.ai.
As it provides many SOTA models for both chat and embeddings. 
This project uses 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo'.
It is not heavy for local deployment and it is particularly good at summarization and code generation. 
It is also efficient more on consumer grade GPU as its context length is not too large as well -> 128K

### 3. Embedding Model:
For embeddings I used 'BAAI/bge-large-en-v1.5' as its popular for its embeddings for tasks like RAG and the workflow is RAG based.

### Contextual Chunking: Here we used contextual chunking which uses LLM to add an extra layer of context to each chunks making it more semanticaly meaningful and it helps enhances the retrieval.

### Reranker: To ensure better retrieval I used cohere's endpoint of Reranker to rank best answer out of top 5
For better result instead of direct returning the answer in retrieval.

### For Data Extractor tool I used few shot prompting as it follows a different output format (JSON).
