import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

from langchain_community.document_compressors import FlashrankRerank
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import ContextualCompressionRetriever
from flashrank import Ranker

# Configuration Defaults
dataDir = os.getenv("DATA_DIR", "./data")
indexBaseDir = os.getenv("INDEX_BASE_DIR", "./faiss_index")
flashrankCacheDir = os.getenv("FLASHRANK_CACHE_DIR", "./flashrank_cache")
ollamaUrl = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
embeddingModel = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

# Default models to compare if none provided
defaultModels = ["gemma2:9b", "qwen2.5:7b", "llama3.1:8b"]

# Medical RAG Prompt Template
clinicalPromptTemplate = """
### MEDICAL RESEARCH & EXTRACTION ASSISTANT ###
You are a factual clinical document analyzer. Answer questions and parse information strictly using ONLY the provided medical context.

### CLINICAL ACCURACY PROTOCOL ###
1. STRICT CITATION: Quote exact numbers, units (e.g., mg, mmol/L, bpm), lab reference ranges, and dosages. Never approximate or guess dosages.
2. CONTEXTUAL CHECK: Distinguish clearly between past medical history, family history, and active diagnoses.
3. NEGATION AWARENESS: Distinguish between affirmed symptoms and ruled-out conditions (e.g., "no evidence of infarction").
4. BOUNDARIES: If the provided text does not explicitly state the answer or the links needed to deduce it, state: "Not documented in the provided medical record." Do not assume or extrapolate clinical diagnoses.

### CONTEXT ###
{context}

### QUESTION / EXTRACTION GOAL ###
Question: {question}

### CLINICAL ANALYSIS & FINAL ANSWER ###
Answer:"""


def getVectorstore(pdfFilename: str, embeddings: OllamaEmbeddings) -> FAISS:
    """Loads or creates a cached FAISS index for a specific PDF."""
    pdfPath = os.path.join(dataDir, pdfFilename)
    indexPath = os.path.join(indexBaseDir, pdfFilename.replace(".", "_"))

    if not os.path.exists(pdfPath):
        raise FileNotFoundError(f"PDF document not found: {pdfPath}")

    if os.path.exists(os.path.join(indexPath, "index.faiss")):
        return FAISS.load_local(
            indexPath, embeddings, allow_dangerous_deserialization=True
        )

    print(f"--- Building new FAISS index for {pdfFilename} ---")
    loader = PyPDFLoader(pdfPath)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            "##",
            "###",
            "Patient:",
            "Assessment:",
            "Plan:",
            "Diagnosis:",
            "History of Present Illness:",
            "Medications:",
            "Allergies:",
            "Lab Results:",
            ". ",
            " ",
            "",
        ],
    )
    splits = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local(indexPath)
    return vectorstore


def evaluateResponse(response: str, testCase: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluates the model response against expected keywords and guardrails."""
    lowerResp = response.lower()

    # Check 1: Expected keywords (all must be present)
    missingKeywords = [
        kw for kw in testCase.get("expected_keywords", [])
        if kw.lower() not in lowerResp
    ]
    keywordsPassed = len(missingKeywords) == 0

    # Check 2: Forbidden keywords (none must be present)
    forbiddenHits = [
        bad for bad in testCase.get("must_not_contain", [])
        if bad.lower() in lowerResp
    ]
    forbiddenPassed = len(forbiddenHits) == 0

    # Check 3: Boundary check adherence
    boundaryPassed = True
    if testCase.get("is_boundary_test", False):
        boundaryPassed = "not documented" in lowerResp

    overallPassed = keywordsPassed and forbiddenPassed and boundaryPassed

    return {
        "passed": overallPassed,
        "keywords_passed": keywordsPassed,
        "missing_keywords": missingKeywords,
        "forbidden_passed": forbiddenPassed,
        "forbidden_hits": forbiddenHits,
        "boundary_passed": boundaryPassed,
    }


def runBenchmark(
    models: List[str],
    datasetPath: str,
    outputJson: str = "benchmark_results.json",
    ollamaUrl: str = ollamaUrl,
):
    print("=" * 80)
    print(" MEDICAL RAG LLM BENCHMARK SUITE")
    print(f" Ollama URL:      {ollamaUrl}")
    print(f" Embedding Model: {embeddingModel}")
    print(f" Candidate Models: {', '.join(models)}")
    print(f" Test Dataset:    {datasetPath}")
    print("=" * 80)

    if not os.path.exists(datasetPath):
        print(f"Error: Dataset file '{datasetPath}' not found.")
        sys.exit(1)

    with open(datasetPath, "r") as f:
        testCases = json.load(f)

    # Initialize embeddings and cache
    embeddings = OllamaEmbeddings(model=embeddingModel, base_url=ollamaUrl)

    # Initialize FlashRank reranker
    flashrankDir = flashrankCacheDir if os.path.exists(flashrankCacheDir) else None
    compressor = FlashrankRerank(
        client=Ranker(model_name="ms-marco-MultiBERT-L-12", cache_dir=flashrankDir)
    )
    prompt = ChatPromptTemplate.from_template(clinicalPromptTemplate)

    # Preload vectorstores to isolate LLM inference time
    vectorstores = {}
    for case in testCases:
        pdf = case["pdf"]
        if pdf not in vectorstores:
            try:
                vectorstores[pdf] = getVectorstore(pdf, embeddings)
            except Exception as e:
                print(f"Failed to load vectorstore for {pdf}: {e}")
                sys.exit(1)

    # Dictionary to store benchmark results
    benchResults = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": {},
    }

    for modelName in models:
        print(f"\n[>>>] Benchmarking Model: {modelName}")
        modelResults = {
            "test_runs": [],
            "total_latency_sec": 0.0,
            "total_passed": 0,
            "total_tests": len(testCases),
            "category_scores": {},
        }

        try:
            llm = ChatOllama(model=modelName, base_url=ollamaUrl, temperature=0)
            # Quick connectivity check
            llm.invoke("Hi")
        except Exception as e:
            print(f"  [ERROR] Could not connect to model '{modelName}'. Is it pulled in Ollama? Error: {e}")
            modelResults["error"] = str(e)
            benchResults["models"][modelName] = modelResults
            continue

        for idx, case in enumerate(testCases, 1):
            category = case.get("category", "General")
            question = case["question"]
            pdf = case["pdf"]

            if category not in modelResults["category_scores"]:
                modelResults["category_scores"][category] = {"passed": 0, "total": 0}
            modelResults["category_scores"][category]["total"] += 1

            vstore = vectorstores[pdf]
            retriever = vstore.as_retriever(search_kwargs={"k": 10})
            compressionRetriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=retriever
            )

            # Retrieve context snippets
            docs = compressionRetriever.invoke(question)
            context = "\n\n".join(d.page_content for d in docs)

            # Run and time LLM generation
            chain = prompt | llm | StrOutputParser()
            startTime = time.perf_counter()
            try:
                response = chain.invoke({"context": context, "question": question})
            except Exception as e:
                response = f"[GENERATION ERROR: {e}]"
            latency = time.perf_counter() - startTime

            # Evaluate response
            evalMetrics = evaluateResponse(response, case)
            if evalMetrics["passed"]:
                modelResults["total_passed"] += 1
                modelResults["category_scores"][category]["passed"] += 1

            modelResults["total_latency_sec"] += latency
            statusSymbol = "✓ PASS" if evalMetrics["passed"] else "✗ FAIL"

            print(
                f"  Test {idx}/{len(testCases)} [{category[:20]:<20}] "
                f"{statusSymbol} | Latency: {latency:5.2f}s"
            )

            if not evalMetrics["passed"]:
                if evalMetrics["missing_keywords"]:
                    print(f"     └─ Missing expected: {evalMetrics['missing_keywords']}")
                if evalMetrics["forbidden_hits"]:
                    print(f"     └─ Found forbidden:  {evalMetrics['forbidden_hits']}")
                if not evalMetrics["boundary_passed"]:
                    print("     └─ Failed boundary check ('Not documented...' missing)")

            modelResults["test_runs"].append({
                "test_id": case.get("id", f"test_{idx}"),
                "category": category,
                "question": question,
                "latency_sec": round(latency, 2),
                "evaluation": evalMetrics,
                "response": response,
            })

        modelResults["avg_latency_sec"] = round(
            modelResults["total_latency_sec"] / len(testCases), 2
        )
        modelResults["accuracy_percent"] = round(
            (modelResults["total_passed"] / len(testCases)) * 100, 1
        )
        benchResults["models"][modelName] = modelResults

    # Print Summary Table
    printSummaryTable(benchResults)

    # Save to JSON
    with open(outputJson, "w") as f:
        json.dump(benchResults, f, indent=2)
    print(f"\n[+] Full benchmark results saved to '{outputJson}'")


def printSummaryTable(benchResults: Dict[str, Any]):
    print("\n" + "=" * 90)
    print(" BENCHMARK SUMMARY REPORT")
    print("=" * 90)
    header = f"{'Model':<20} | {'Pass Rate':<12} | {'Avg Latency':<14} | {'Total Time':<12} | {'Status'}"
    print(header)
    print("-" * len(header))

    for model, res in benchResults["models"].items():
        if "error" in res:
            print(f"{model:<20} | {'N/A':<12} | {'N/A':<14} | {'N/A':<12} | ERROR")
            continue

        passRate = f"{res['total_passed']}/{res['total_tests']} ({res['accuracy_percent']}%)"
        avgLat = f"{res['avg_latency_sec']}s / query"
        totTime = f"{res['total_latency_sec']:.2f}s"
        status = "PASSED" if res["accuracy_percent"] >= 80.0 else "SUBPAR"

        print(f"{model:<20} | {passRate:<12} | {avgLat:<14} | {totTime:<12} | {status}")

    print("=" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark Ollama models for Medical RAG assistant"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=defaultModels,
        help="List of Ollama models to benchmark (e.g. --models gemma2:9b qwen2.5:7b llama3.1:8b)",
    )
    parser.add_argument(
        "--dataset",
        default="eval_dataset.json",
        help="Path to evaluation dataset JSON file",
    )
    parser.add_argument(
        "--output",
        default="benchmark_results.json",
        help="Path to output results JSON file",
    )
    parser.add_argument(
        "--ollama-url",
        default=ollamaUrl,
        help="Ollama base URL (defaults to OLLAMA_BASE_URL env var)",
    )

    args = parser.parse_args()
    runBenchmark(
        models=args.models,
        datasetPath=args.dataset,
        outputJson=args.output,
        ollamaUrl=args.ollama_url,
    )
