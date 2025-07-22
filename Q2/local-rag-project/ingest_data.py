# ingest_data.py

import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# --- Configuration ---
# Define the directories for the knowledge base and the vector store.
# Using os.path.join ensures compatibility across different operating systems.
KNOWLEDGE_BASE_DIR = "knowledge_base"
VECTOR_STORE_DIR = "vector_stores"
INTENT_CATEGORIES = ["technical", "billing", "feature_requests"]

def ingest_data():
    """
    Ingests data from the knowledge base, processes it, and stores it in a vector database.
    This function creates a separate vector store for each intent category.
    """
    print("Starting data ingestion process...")

    # Initialize the embedding model we'll use. We're using a local Ollama model.
    embeddings = OllamaEmbeddings(model="llama3", show_progress=True)

    # The text splitter will break down large documents into smaller chunks.
    # This is crucial for the RAG model to find specific, relevant information.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=100)

    for intent in INTENT_CATEGORIES:
        intent_kb_path = os.path.join(KNOWLEDGE_BASE_DIR, intent)
        intent_vs_path = os.path.join(VECTOR_STORE_DIR, f"chroma_db_{intent}")

        if not os.path.exists(intent_kb_path):
            print(f"Warning: Directory not found for intent '{intent}': {intent_kb_path}")
            continue

        print(f"--- Processing intent: {intent} ---")
        print(f"Loading documents from: {intent_kb_path}")

        # Load all .txt documents from the specific intent directory
        loader = DirectoryLoader(intent_kb_path, glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()

        if not documents:
            print(f"No documents found for intent '{intent}'. Skipping.")
            continue

        # Split the loaded documents into manageable chunks
        splits = text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(splits)} chunks.")

        # Create the vector store from the document chunks.
        # This will convert each chunk to an embedding and store it.
        # The `persist_directory` argument tells ChromaDB to save the data to disk.
        print(f"Creating vector store at: {intent_vs_path}")
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=intent_vs_path
        )
        print(f"Successfully created vector store for '{intent}'.")

    print("--- Data ingestion complete. ---")


if __name__ == "__main__":
    # This block allows you to run the script directly from the command line.
    ingest_data()