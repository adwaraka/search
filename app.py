import os
import sys

from langchain_ollama import (
    OllamaEmbeddings,
    ChatOllama,
)
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from flashrank import Ranker

# Config
DATA_DIR = "./data"
INDEX_BASE_DIR = "./faiss_index"
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "gemma2:9b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

# Models - Gemma 2 for high accuracy and clinical reasoning
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_URL)
llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_URL, temperature=0)


def getVectorstore(pdf_filename):
    """
    Checks if a FAISS index exists for a specific PDF. 
    If yes, loads it. If no, creates and saves it.
    """
    pdf_path = os.path.join(DATA_DIR, pdf_filename)
    # Create a unique directory name for this specific PDF's index
    index_path = os.path.join(INDEX_BASE_DIR, pdf_filename.replace(".", "_"))
    
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_filename} not found in {DATA_DIR}")
        return None

    if os.path.exists(os.path.join(index_path, "index.faiss")):
        print(f"--- Loading cached index for {pdf_filename} ---")
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    
    print(f"--- Building new index for {pdf_filename} (this may take a minute) ---")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Clinical text splitting with section-aware separators
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
    vectorstore.save_local(index_path)
    return vectorstore


def runRagChat(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    # Strict prompt for medical parsing & clinical extraction accuracy
    template = """
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
    prompt = ChatPromptTemplate.from_template(template)

    def formatDocs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


    def debugDocuments(docs):
        # CRITICAL CHECK: If it's a string, we've already formatted it. 
        # We can't debug metadata of a string.
        if isinstance(docs, str):
            return docs 

        print(f"\n[DEBUG] Re-ranker selected {len(docs)} documents:")
        for i, doc in enumerate(docs):
            # Flashrank sometimes flattens metadata; we use .get() to be safe
            page = doc.metadata.get('page', 'N/A')
            score = doc.metadata.get('relevance_score', 'N/A')
            snippet = doc.page_content[:60].replace('\n', ' ')
            print(f"  {i+1}. Page {page} | Rel-Score: {score} | Snippet: {snippet}...")
        
        return docs


    # By pulling 10 results from FAISS but using a Re-ranker to pick the best 5
    compressor = FlashrankRerank(
        client=Ranker(model_name="ms-marco-MultiBERT-L-12", cache_dir="/app/flashrank_cache")
        )
    compressionRetriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever
    )
    chain = (
        {
            "context": compressionRetriever | debugDocuments | formatDocs,
            "question": RunnablePassthrough()
         }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n--- CHAT READY (Type 'exit' to switch files or quit) ---")
    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            break

        response = chain.invoke(query)
        print(f"\nAI: {response}")


if __name__ == "__main__":
    while True:
        availableFiles = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
        print()
        for index, file in enumerate(availableFiles):
            print(f"{index} :  {file}")
        print()

        fileChoice = input(
            "Enter PDF filename or select the number (or 'exit'): "
            ).strip()

        if fileChoice.lower() == "exit":
            break

        try:
            # document index has been selected
            documentNumber = int(fileChoice)
            vstore = getVectorstore(availableFiles[documentNumber])
        except ValueError:
            vstore = getVectorstore(fileChoice)

        if vstore:
            runRagChat(vstore)
