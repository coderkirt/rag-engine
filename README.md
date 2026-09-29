# rag-engine

A hybrid-retrieval RAG (Retrieval-Augmented Generation) engine that answers questions over your own PDFs/TXT documents — built with FastAPI, LangChain, ChromaDB and Groq-hosted LLMs, and paired with an **LLM-as-a-Judge evaluation harness** including negative (refusal) test cases.

> **Why this project exists:** most RAG demos stop at "chat with your PDF". This one focuses on the parts that usually get skipped — grounding answers in sources, refusing to answer when the corpus doesn't contain the answer, and *measuring* retrieval + answer quality with an automated test suite.

---

## Features

- **Hybrid retrieval** — dense semantic search (Chroma + `all-MiniLM-L6-v2`) combined with sparse BM25 keyword search (`rank_bm25`), merged and deduplicated to catch both semantic and exact-term matches.
- **Automatic document categorization** — every ingested file is classified by an LLM into a short topic tag (e.g. *"Marine Plastic Detection"*); tags are cached in `category_cache.json` so files are only classified once.
- **Metadata-aware filtering** — restrict retrieval to a single file or category at query time.
- **Grounded answers with source metadata** — every response returns the `file_name` and `category` of the chunks used, so answers are auditable.
- **Refusal behavior** — when the corpus doesn't contain the answer, the assistant says so instead of hallucinating.
- **LLM-as-a-Judge evaluation harness** — a 10-case suite (direct answers, negative/refusal cases, cross-domain synthesis questions) scored by a second LLM, with a strict "no sources retrieved ⇒ automatic fail" grounding guard and JSON audit logging.
- **REST API + lightweight UI** — FastAPI backend with document upload and DB reset endpoints; Streamlit chat frontend.

---

## Architecture

```
                        ┌──────────────────────────────────────────────┐
                        │                 INGESTION                    │
  app/data/*.pdf,*.txt  │  DirectoryLoader → RecursiveCharacterSplitter│
 ─────────────────────► │  (chunk 1000 / overlap 200)                  │
                        │        │                                     │
                        │        ▼                                     │
                        │  LLM auto-categorizer (cached per file)      │
                        └────────┬─────────────────────────────────────┘
                                 ▼
                 ┌───────────────────────────────┐
                 │  ChromaDB (dense vectors)     │◄── all-MiniLM-L6-v2
                 │  BM25 index (sparse, rebuilt  │
                 │  from in-memory chunks)       │
                 └───────────────┬───────────────┘
                                 ▼
        query ─► both retrievers (top-5 each) ─► merge + dedupe ─► top-3 context
                                 ▼
                 Groq LLM (prompted to answer ONLY from context)
                                 ▼
                 { answer, metadata: [source file + category per chunk] }
```

**Evaluation flow:** `qa_suite/test_cases.json` → `evaluator.py` runs each case through the engine → a judge LLM scores the actual vs. expected answer (pass ≥ 8/10) → grounding guard + negative-case checks → `qa_audit_report.json`.

---

## Project structure

```
rag-engine/
├── app/
│   ├── core/
│   │   ├── logger.py            # Append-only JSON audit log for eval runs
│   │   └── validator.py         # LLM-as-a-Judge (OpenAI-compatible API)
│   ├── services/
│   │   └── rag_engine.py        # Core engine: ingestion, hybrid retrieval, QA
│   ├── data/                    # ⬅ drop your .pdf / .txt files here (gitignored)
│   └── main.py                  # FastAPI application
├── qa_suite/
│   └── test_cases.json          # Eval suite: positive, negative & cross-domain cases
├── evaluator.py                 # Runs the eval harness → qa_audit_report.json
├── qa_evaluator.py              # Secondary faithfulness/relevance audit pass
├── frontend_app.py              # Streamlit chat UI (upload, reset, chat)
└── test_*.py                    # Manual smoke scripts (db, loader, hybrid, rag)
```

---

## Quick start

**Requirements:** Python 3.10+, a free [Groq API key](https://console.groq.com/keys).

```bash
# 1. Clone and set up the environment
git clone https://github.com/upadhyayaditya7/rag-engine.git
cd rag-engine
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install fastapi "uvicorn[standard]" python-multipart streamlit requests python-dotenv \
    langchain-community langchain-text-splitters langchain-huggingface langchain-chroma \
    langchain-groq rank_bm25 sentence-transformers pypdf openai

# 3. Configure your API key
echo "GROQ_API_KEY=your_key_here" > .env

# 4. Check the LLM model name
#    The model is hardcoded in app/services/rag_engine.py. Groq retires models
#    periodically — if /query returns a 500 with "model does not exist", update
#    the model_name in RAGEngine.__init__ to one your key can access
#    (see https://console.groq.com/docs/models).

# 5. Add documents
#    Place at least one .pdf or .txt into app/data/

# 6. Start the backend
uvicorn app.main:app --reload

# 7. (optional) Start the chat UI in a second terminal
streamlit run frontend_app.py
```

The first startup ingests and indexes every document in `app/data/` (one LLM call per file for categorization, then embedding). Subsequent startups reuse the persisted Chroma DB and the category cache.

### Ask a question

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What augmentation techniques were used in the marine plastic study?"}'
```

```json
{
  "answer": "The study used flipping, rotating, colour correction ...",
  "metadata": [
    { "file_name": "Marine_Plastic_Detection_Using_Deep_Learning.pdf", "category": "Marine Plastic Detection", "...": "..." }
  ]
}
```

### API endpoints

| Method | Path | Description |
|---|---|---|
| `GET`  | `/` | Health check |
| `POST` | `/query` | Ask a question — body: `{"question": str, "session_id": str}` |
| `POST` | `/api/upload` | Upload + index a document (multipart form field `file`, `.pdf`/`.txt`) |
| `POST` | `/api/reset-database` | Wipe the vector DB and re-index `app/data/` |

---

## Running the evaluation harness

```bash
python evaluator.py
```

For each case in `qa_suite/test_cases.json` the harness:

1. Queries the engine (with category filters for document-specific questions, unfiltered for cross-domain ones);
2. Applies a **grounding guard** — a standard question that retrieves zero sources fails automatically;
3. Verifies **negative cases** — questions about things *not* in the corpus must be refused, not answered;
4. Scores standard cases with an **LLM-as-a-Judge** (`EVAL_MODEL`, default `llama-3.3-70b-versatile`), passing at ≥ 8/10;
5. Appends every result to `qa_audit_report.json` for auditing.

---

## Known limitations

This is a portfolio/learning project — honest about what it is and isn't:

- **The LLM model name is hardcoded** rather than configured via `.env`. Groq has retired model IDs before (e.g. `llama-3.1-8b-instant`), which turns every query into a 500 until the constant is updated. Pin it to a model available to your key, or move it to an environment variable.
- **No authentication.** Anyone who can reach the API can upload files or wipe the database, and CORS is wide open. Do not expose it publicly as-is.
- **Upload filenames are not sanitized** (a `../` filename can escape the data directory) and file size is unbounded. Fix before any multi-user use.
- **Empty corpus crashes startup.** With no documents in `app/data/`, ingestion raises an empty-upsert error from Chroma. Always ingest ≥ 1 document.
- **Full re-ingestion on every upload/reset** — documents are re-split and re-embedded rather than indexed incrementally. Fine for a handful of PDFs, not for thousands.
- **Single-turn only.** `session_id` is accepted but no conversation memory is implemented yet.
- **Naive BM25 tokenization** (whitespace split, lowercased) — punctuation and stopwords are not handled.
- **LLM-as-a-Judge has known biases** (verbosity, self-preference). The harness mitigates this with the grounding guard and refusal checks, but scores should be read as directional, not absolute.

## Roadmap

- [ ] Sanitize uploads (basename + extension allowlist + size cap) and add API-key auth
- [ ] Incremental indexing (per-file) instead of full re-ingestion
- [ ] Multi-turn session memory behind `session_id`
- [ ] Pytest suite with mocked LLM + GitHub Actions CI
- [ ] Reranker stage and configurable retriever weights

---

## License

[Apache-2.0](LICENSE)
