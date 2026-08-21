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

# Config
DATA_DIR = "./data"
INDEX_BASE_DIR = "./faiss_index"
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

# Models - Using 3B for a balance of speed and logic
embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_URL)
llm = ChatOllama(model="llama3.2:3b", base_url=OLLAMA_URL, temperature=0)


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

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local(index_path)
    return vectorstore


def runRagChat(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    # Stricter prompt for better accuracy
    template = """
    ### SYSTEM INSTRUCTIONS ###
    You are a factual research assistant and logical analyst.
    Answer the question using ONLY the provided context below.

    ### LOGICAL PROTOCOL ###
    Before providing the final answer, perform these internal steps:
    1. IDENTIFY: List specific facts from the context related to the question.
    2. CONNECT: If facts are in different sources, explain how they relate (e.g., Character A's weapon vs. Character B's armor).
    3. CONCLUDE: Provide the answer based strictly on those identified links.

    If the context does not contain the answer or the links needed to deduce it, state "I do not know based on the provided text."

    ### CONTEXT ###
    {context}

    ### QUESTION ###
    Question: {question}

    ### STEP-BY-STEP ANALYSIS & FINAL ANSWER ###
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
    compressor = FlashrankRerank()
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
