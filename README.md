# Medical Document RAG & Parsing Assistant

A privacy-focused, fully local Retrieval-Augmented Generation (RAG) system for parsing, extracting, and querying medical documents and clinical records using **Gemma 2 (9B)**, **LangChain**, **FAISS**, and **FlashRank**.

All processing runs locally on your machine, ensuring patient data and Protected Health Information (PHI) never leave your infrastructure.

---

## Architecture & Hardware Acceleration

```
┌─────────────────────────────────────────────────────────┐
│                     Host Machine (macOS)                │
│                                                         │
│   ┌─────────────────────────────────────────────────┐   │
│   │ Native Ollama (Metal GPU Acceleration)          │   │
│   │ • gemma2:9b (~5.4 GB VRAM)                      │   │
│   │ • nomic-embed-text                              │   │
│   │ Listening on: 127.0.0.1:11434                   │   │
│   └────────────────────────▲────────────────────────┘   │
│                            │                            │
│                 http://host.docker.internal:11434       │
│                            │                            │
│   ┌────────────────────────┴────────────────────────┐   │
│   │ Docker Container: `rag-app`                     │   │
│   │ • LangChain + FAISS Vector Store                │   │
│   │ • FlashRank Re-ranker                           │   │
│   │ • Interactive Clinical Assistant CLI            │   │
│   └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

> **Why Native Ollama?**
> Running Ollama inside Docker on macOS prevents it from accessing Apple Silicon GPU (Metal) acceleration, resulting in slow CPU-only inference. Running Ollama natively on macOS gives it direct access to Apple Silicon GPU / Unified Memory, while keeping the Python RAG application fully containerized in Docker.

---

## Key Features

- **High-Performance Local LLM**: Powered by Google Gemma 2 (`gemma2:9b`) running via native Ollama with Metal GPU acceleration.
- **Clinical-Aware Text Splitting**: Optimized chunking that respects medical sections (Patient info, SOAP notes, Diagnoses, Medications, Dosages, and Lab Results).
- **Fast Vector Retrieval + Re-ranking**: FAISS vector indexing combined with FlashRank reranking for high-precision snippet retrieval.
- **Index Caching**: Persists FAISS indexes locally in `faiss_index/` so documents only need to be embedded once.
- **Strict Clinical Accuracy Prompting**: Built-in guardrails against hallucinations, enforcing exact metric citations, negation handling (e.g., "no evidence of..."), and clear boundary checks.

---

## Directory Structure

```
├── data/               # Place your medical PDF documents here
├── faiss_index/        # Cached FAISS vector indexes (auto-generated)
├── app.py              # Main interactive RAG chat application
├── docker-compose.yml  # Docker compose config (connects to host Ollama)
├── Dockerfile          # Python application container definition
├── requirements.txt    # Python dependencies
└── README.md           # Documentation
```

---

## Prerequisites

- [Docker Desktop for Mac](https://docs.docker.com/desktop/install/mac-install/)
- [Ollama](https://ollama.com) installed natively on macOS (`brew install ollama`)
- Machine with at least 8–16 GB Unified Memory (M1/M2/M3/M4)

---

## Quick Start

### 1. Start Ollama and Pull Models (Host macOS)
In your macOS terminal:
```bash
# Start Ollama service (or open the Ollama macOS App)
ollama serve

# Pull Gemma 2 9B and the embedding model:
ollama pull gemma2:9b
ollama pull nomic-embed-text
```

### 2. Add Medical PDF Documents
Copy any clinical PDFs (discharge summaries, lab results, pathology reports) into the `data/` folder:
```bash
cp /path/to/patient_record.pdf data/
```

### 3. Build and Run the Docker Container
Run the interactive RAG application inside Docker:
```bash
docker compose build && docker compose run --rm rag-app
```

---

## Configuration & Environment Variables

You can customize models and settings in [docker-compose.yml](file:///Users/adwaraka/Desktop/search/docker-compose.yml) or via environment variables:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `OLLAMA_MODEL` | `gemma2:9b` | LLM model tag (`gemma2:9b`, `gemma2:2b`, `gemma2:27b`). |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model tag for document indexing. |
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | Endpoint connecting the container to host Ollama. |

> **Memory Tip for Lower-Spec Hardware:** If running low on memory, switch to `gemma2:2b` (`ollama pull gemma2:2b` and set `OLLAMA_MODEL=gemma2:2b`) for lower memory footprint.

---

## Example Usage Session

```text
0 :  patient_discharge_summary.pdf
1 :  cardiology_report.pdf

Enter PDF filename or select the number (or 'exit'): 0
--- Building new index for patient_discharge_summary.pdf (this may take a minute) ---

--- CHAT READY (Type 'exit' to switch files or quit) ---

You: What medications and dosages were prescribed upon discharge?

[DEBUG] Re-ranker selected 5 documents:
  1. Page 3 | Rel-Score: 0.9421 | Snippet: Discharge Medications: 1. Lisinopril 10 mg PO daily...
  2. Page 4 | Rel-Score: 0.8812 | Snippet: Plan & Follow-up: Continue Atorvastatin 40 mg daily at b...

AI: Based on the discharge summary, the following medications and dosages are documented:
1. Lisinopril: 10 mg PO daily
2. Atorvastatin: 40 mg PO daily at bedtime
3. Metformin: 500 mg PO BID with meals

You: Is there any history of myocardial infarction?

[DEBUG] Re-ranker selected 5 documents:
  1. Page 1 | Rel-Score: 0.9103 | Snippet: Past Medical History: Hypertension, Type 2 Diabetes. No...

AI: No. The medical record explicitly notes: "Past Medical History: Hypertension, Type 2 Diabetes. No prior history of myocardial infarction."

You: exit
```

---

## Benchmarking Alternative LLMs

You can benchmark multiple Ollama models on extraction accuracy, negation handling, boundary enforcement, and query latency using [benchmark.py](file:///Users/adwaraka/Desktop/search/benchmark.py) and [eval_dataset.json](file:///Users/adwaraka/Desktop/search/eval_dataset.json).

### 1. Pull candidate models in Ollama (Host macOS)
```bash
ollama pull gemma2:9b
ollama pull qwen2.5:7b
ollama pull llama3.1:8b
```

### 2. Run the benchmark suite via Docker
```bash
docker compose run --rm rag-app python benchmark.py --models gemma2:9b qwen2.5:7b llama3.1:8b
```

Results and latency metrics will be printed as a summary table and exported to `benchmark_results.json`.

---

## Disclaimer

This tool is designed for document search, data extraction, and research assistance. It is **not** a diagnostic medical device and should not replace professional medical judgment or direct clinical review.