# SummarAI — RAG-Based Document Summarization

Summarize long documents using a Map-Reduce RAG pipeline. Upload any PDF or TXT file and get a structured summary with executive overview, key points, and topics covered.

## How It Works
```
Document → Chunk → FAISS Index → Retrieve Top-K Sections
                                          ↓
                              MAP: Summarize each chunk
                                          ↓
                           REDUCE: Combine into final JSON
                                          ↓
              { executive_summary, key_points, topics_covered }
```

## Tech Stack

- Python
- Groq API (Free — no credit card needed)
- LangChain + LangChain Text Splitters
- FAISS (local vector store)
- HuggingFace all-MiniLM-L6-v2 (local embeddings)
- Streamlit

## Features

- Upload single or multiple PDF and TXT files
- Map-Reduce summarization pipeline
- Structured JSON output with four fields
- One-line summary, executive summary, key points, topics
- Multi-document master cross-document overview
- Chunk inspector to see exactly what the model used
- Download summary as JSON file

## Files

- `app.py` — Streamlit web application
- `SummarAI_RAG.ipynb` — Step-by-step notebook with all pipeline stages
- `requirements.txt` — All dependencies

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Get Free API Key

https://console.groq.com

## Output Format
```json
{
  "one_line_summary": "...",
  "executive_summary": "...",
  "key_points": ["...", "...", "..."],
  "topics_covered": ["...", "...", "..."]
}
```

## My RAG Projects

| Project | Repository |
|---|---|
| Single Document Q&A | https://github.com/hari12314/single-doc-qa-rag |
| Multi Document Q&A | https://github.com/hari12314/Multi-Document-Q-A-System |
| RAG Summarization | https://github.com/hari12314/RAG-Summarization |
