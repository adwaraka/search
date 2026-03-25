import os
import sys

# Modern Imports
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

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
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Stricter prompt for better accuracy
    template = """
    ### SYSTEM INSTRUCTIONS ###
    You are a factual research assistant. Answer the question using ONLY the provided context below.
    The context is your ONE AND ONLY source of truth. If the information is not explicitly mentioned
    in the context, state "I do not know based on the provided text."

    Pay meticulous attention to names of people, objects like weapons, titles, and relationships
    between multiple characters.

    ### CONTEXT ###
    Context:
    {context}

    ## QUESTION ###
    Question: {question}

    ### FINAL ANSWER ###
    Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n--- CHAT READY (Type 'exit' to switch files or quit) ---")
    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            break

        # We use similarity_search_with_score to see the 'how'
        docs_with_scores = vectorstore.similarity_search_with_score(query, k=2)
        print(f"\n[DEBUG] Top Source: Page {docs_with_scores[0][0].metadata.get('page')} (Score: {docs_with_scores[0][1]:.4f})")

        response = chain.invoke(query)
        print(f"\nAI: {response}")


if __name__ == "__main__":
    while True:
        available_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
        print(f"\nAvailable PDFs: {available_files}")        
        file_choice = input("Enter PDF filename (or 'exit'): ").strip()

        if file_choice.lower() == "exit":
            break

        vstore = getVectorstore(file_choice)
        if vstore:
            runRagChat(vstore)
