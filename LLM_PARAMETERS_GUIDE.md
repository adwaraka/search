# LLM Benchmarking & Parameter Gauging Guide

This document details the critical parameters to gauge when benchmarking, evaluating, and configuring Large Language Models (LLMs) via Ollama, LangChain, and RAG pipelines for clinical document search and extraction.

---

## 1. Inference & Sampling Hyperparameters

These parameters govern the token selection mathematics during text generation.

| Parameter | Type / Range | What It Determines | Impact on Medical RAG & Extraction | Recommended Value |
| :--- | :--- | :--- | :--- | :--- |
| **`temperature`** | Float (`0.0` - `2.0`) | Degree of randomness in next-token probability distribution. | At `0.0` (greedy search), the model selects the single highest-probability token every time. Crucial for reproducible extraction; higher values introduce hallucinations in dosages and lab metrics. | **`0.0`** *(Strictly deterministic)* |
| **`top_p`** (Nucleus Sampling) | Float (`0.0` - `1.0`) | Cumulative probability threshold for candidate tokens. | Truncates the probability tail. Prevents the model from picking rare or nonsensical tokens when extracting medical data. | **`0.1` - `0.3`** *(or `1.0` when `temperature=0.0`)* |
| **`top_k`** | Integer (`1` - `100+`) | Hard limit on the pool of highest-probability tokens considered. | Similar to `top_p`, limits sampling diversity. Lower values keep extraction tightly bound to common medical terms. | **`10` - `40`** |
| **`repeat_penalty`** | Float (`1.0` - `2.0`) | Multiplier penalty against generating tokens that already appeared in the prompt/response. | **Warning in Medical RAG:** Setting this too high (`> 1.15`) forces the model to find synonyms rather than repeating exact drug names (*Lisinopril*), units (*mg/dL*), and clinical terms. | **`1.0` - `1.05`** |
| **`presence_penalty`** | Float (`-2.0` - `2.0`) | Flat penalty applied once if a token has appeared in the text. | Encourages introducing new topics. Should remain neutral (`0.0`) for strict Q&A extraction. | **`0.0`** |
| **`frequency_penalty`** | Float (`-2.0` - `2.0`) | Proportional penalty scaling with the number of times a token was repeated. | Prevents repetitive loops. Should remain neutral to avoid penalizing recurring units (e.g. repeated `mg` in medication lists). | **`0.0`** |
| **`num_predict`** (`max_tokens`) | Integer (`1` - `4096+`) | Maximum number of new tokens the model is permitted to generate. | Caps response length, prevents infinite generation loops, and places an upper bound on response latency. | **`256` - `512`** |
| **`stop`** | List of Strings | Sequence strings that immediately halt token generation when encountered. | Ensures clean generation boundaries (e.g., stopping at `"\n\nQuestion:"` or `"###"`). | `["\n\nQuestion:", "###"]` |

---

## 2. Context & RAG Architecture Parameters

These parameters control how much document text is fed to the model and how retrieval is orchestrated.

| Parameter | Default in Ollama | What It Determines | Impact / Gotcha | Recommended Value |
| :--- | :--- | :--- | :--- | :--- |
| **`num_ctx`** (Context Window) | `2048` tokens | Maximum token capacity allocated in RAM/VRAM for system prompt + retrieved context + user question + answer. | **Critical Pitfall:** Ollama defaults to only 2048 tokens. If 5 FlashRank context chunks + prompt exceed 2048 tokens, Ollama **silently truncates** context, leading to missed facts and false hallucinations without throwing an error. | **`8192`** or **`16384`** |
| **`chunk_size`** | N/A (Text Splitter) | Number of characters per document chunk. | Smaller chunks (`400-600`) preserve granular facts; larger chunks (`800-1200`) preserve clinical narratives and table relationships. | **`800`** *(with section separators)* |
| **`chunk_overlap`** | N/A (Text Splitter) | Overlapping characters between sequential chunks. | Prevents cutting a vital dosage or sentence across a chunk boundary. | **`150`** |
| **Retriever Top-$K$ (`search_kwargs={"k": N}`)** | `4` | Number of initial candidate chunks pulled from FAISS vector search. | High initial $K$ ensures high recall before passing to the reranker. | **`10`** |
| **Re-ranker Depth (`FlashRank`)** | N/A | Number of top re-scored chunks passed to the LLM context. | Balances prompt token count with information density. Reduces noise and "lost-in-the-middle" attention degradation. | **`3` - `5`** |

---

## 3. Quantization & Hardware Execution Parameters

These parameters control memory footprint, precision, and Apple Silicon Metal GPU acceleration.

| Parameter | Options | What It Determines | Trade-offs in Medical Search |
| :--- | :--- | :--- | :--- |
| **Quantization Precision** | `Q4_K_M`, `Q5_K_M`, `Q8_0`, `FP16` | Bit-width reduction of model weights. | • **`Q4_K_M` (4-bit)**: Minimal memory (~5.4 GB for 9B), fastest generation. May occasionally exhibit minor precision drift on decimal lab numbers (e.g., `<0.01`).<br>• **`Q8_0` (8-bit)**: Near full precision, highest extraction reliability for numerical clinical data, but requires ~9.5 GB VRAM for 9B models. |
| **`num_gpu` / GPU Layers** | Integer (`0` to all layers) | Number of neural network layers offloaded to GPU memory. | On Apple Silicon (macOS Metal), offloading 100% of layers to Unified Memory ensures 30–60+ tokens/sec. Partial CPU offload drops throughput to 2–5 tokens/sec. |
| **`num_thread`** | Integer (CPU cores) | Number of CPU worker threads used during prompt evaluation if CPU is utilized. | Set to match physical CPU performance cores when GPU offload is unavailable. |
| **`use_mmap`** | Boolean (`true`/`false`) | Memory-maps model files directly into RAM. | Enables fast startup and sharing across processes; default is `true`. |
| **`use_mlock`** | Boolean (`true`/`false`) | Locks model weights in RAM, preventing macOS from paging weights to SSD swap. | Keeps latency consistent; useful on machines with constrained Unified Memory. |

---

## 4. Operational Metrics Gauged During Benchmarking

When testing combinations of models and parameters, measure these 5 key dimensions:

```
┌──────────────────────────────────────┬─────────────────────────────────────────────────────────────┐
│ Metric                               │ Definition & Significance                                   │
├──────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 1. Time to First Token (TTFT)        │ Time (seconds) spent processing and ingesting the prompt   │
│                                      │ and retrieved context before generating the first token.    │
├──────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 2. Tokens Per Second (TPS)           │ Rate of token generation during answer synthesis.          │
│                                      │ Reflects interactive responsiveness for clinicians.         │
├──────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 3. Peak VRAM / Memory Footprint      │ Total unified memory consumed by Ollama + LangChain app.    │
│                                      │ Must fit within system limits without triggering swap.     │
├──────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 4. Extraction Accuracy Rate          │ Percentage of required clinical entities (names, dosages,   │
│                                      │ units) extracted with exact character/numeric fidelity.     │
├──────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ 5. Guardrail & Boundary Compliance   │ Rate at which absent data triggers "Not documented..."      │
│                                      │ and negative findings are correctly recognized.             │
└──────────────────────────────────────┴─────────────────────────────────────────────────────────────┘
```

---

## 5. Python / LangChain Configuration Reference

Here is how all gauged parameters are passed when configuring `ChatOllama`:

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="gemma2:9b",
    base_url="http://host.docker.internal:11434",
    
    # --- Sampling & Determinism ---
    temperature=0.0,            # 0.0 = Deterministic greedy decoding
    top_p=0.2,                  # Low nucleus threshold to filter noise
    top_k=20,                   # Restrict token selection pool
    repeat_penalty=1.05,        # Minimal penalty to allow repeated medical units
    
    # --- Context & Capacity ---
    num_ctx=8192,               # Prevent silent context truncation (default is 2048)
    num_predict=512,            # Max generated tokens
    
    # --- Performance ---
    num_gpu=99,                 # Ensure 100% layer offload to Metal GPU
    stop=["\n\nQuestion:", "###"]
)
```

---

## 6. Ollama Modelfile Parameter Reference

If you create custom Ollama models using a `Modelfile`, you can bake these parameters in permanently:

```dockerfile
FROM gemma2:9b

# Set default parameter values
PARAMETER temperature 0.0
PARAMETER top_p 0.2
PARAMETER top_k 20
PARAMETER repeat_penalty 1.05
PARAMETER num_ctx 8192
PARAMETER num_predict 512

# Set clinical system prompt
SYSTEM """
You are a factual clinical document analyzer. Answer questions strictly using ONLY the provided medical context. Quote exact numbers, units, and dosages. If information is absent, state: 'Not documented in the provided medical record.'
"""
```
