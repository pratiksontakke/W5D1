# app/chains.py

import os
from langchain_ollama import ChatOllama, OllamaEmbeddings  # Use langchain_ollama for both
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough, RunnableLambda

# --- Configuration ---
VECTOR_STORE_DIR = "vector_stores"

# --- Model Selection Logic ---
def get_llm(streaming: bool = False):
    """Returns an instance of the local Ollama LLM (qwen3:0.6b)."""
    return ChatOllama(
        model="qwen3:0.6b",
        temperature=0,
        streaming=streaming,
        format="json" if not streaming else None,
        base_url="http://localhost:11434"
    )

# --- Embeddings (Using local Ollama qwen3:0.6b) ---
embeddings = OllamaEmbeddings(
    model="qwen3:0.6b",
    base_url="http://localhost:11434"
)

# --- Helper function to create RAG chains ---
def create_rag_chain(intent: str, streaming: bool = False):
    """Creates a Retrieval-Augmented Generation (RAG) chain for a specific intent."""
    rag_prompt_template = """
    You are a helpful and friendly customer support assistant for a SaaS company.
    Answer the user's question based ONLY on the context provided below.
    If the context does not contain the answer, state that you do not have enough information to answer.
    Do not make up information.

    <context>
    {context}
    </context>

    Question: {question}
    Answer:
    """
    rag_prompt = PromptTemplate.from_template(rag_prompt_template)

    # Load the specific vector store for the given intent
    vector_store_path = os.path.join(VECTOR_STORE_DIR, f"chroma_db_{intent.lower().replace(' ', '_')}")
    vector_store = Chroma(persist_directory=vector_store_path, embedding_function=embeddings)
    retriever = vector_store.as_retriever()

    # Get LLM instance with appropriate streaming setting
    llm = get_llm(streaming=streaming)

    # The RAG chain pipeline
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# --- Chain: Intent Classifier Chain ---
def merge_intent_and_query(input_dict, intent_result):
    return {**input_dict, **intent_result}

def get_full_chain(streaming: bool = False):
    """
    Builds and returns the full router chain with streaming support.
    """
    # Use a dedicated LLM for classification (non-streaming, JSON format)
    classifier_llm = get_llm(streaming=False)
    
    classifier_prompt_template = """
    You are an expert intent classifier for a SaaS company's support system.
    Your task is to categorize the user's query into one of the following predefined categories.
    Respond ONLY with a JSON object containing the key "intent" and the determined category.

    <categories>
    - "Technical Support": For questions about API usage, bugs, errors, password resets, how-to guides, and integration problems.
    - "Billing/Account": For questions about pricing, subscriptions, invoices, payment methods, and account limits.
    - "Feature Request": For suggestions for new features, improvements to existing features, or product feedback.
    - "General Inquiry": For all other questions that do not fit into the above categories.
    </categories>

    Classify the following user query:
    User Query: {query}
    """
    classifier_prompt = PromptTemplate.from_template(classifier_prompt_template)
    intent_chain = (
        classifier_prompt
        | classifier_llm
        | JsonOutputParser()
        | RunnableLambda(lambda out, input: {**input, **out})
    )

    # Create RAG chains with streaming support
    technical_rag_chain = create_rag_chain("technical", streaming)
    billing_rag_chain = create_rag_chain("billing", streaming)
    feature_rag_chain = create_rag_chain("feature_requests", streaming)
    
    # Fallback chain for general inquiries
    general_inquiry_prompt = PromptTemplate.from_template(
        "You are a general support assistant. Answer the following question as best you can: {query}"
    )
    general_inquiry_chain = general_inquiry_prompt | get_llm(streaming=streaming) | StrOutputParser()

    return intent_chain | RunnableBranch(
        (lambda x: x.get("intent") == "Technical Support", technical_rag_chain),
        (lambda x: x.get("intent") == "Billing/Account", billing_rag_chain),
        (lambda x: x.get("intent") == "Feature Request", feature_rag_chain),
        general_inquiry_chain
    )
