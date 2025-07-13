from dotenv import load_dotenv
import os

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import Chroma

from pydantic import SecretStr
from ragas import evaluate
from ragas.metrics import faithfulness
from datasets import Dataset

print("Initializing RAG system components...")

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables")

llm = OpenAI(
    model="gpt-3.5-turbo-instruct",
    temperature=0.7,
    max_retries=2,
    api_key=SecretStr(openai_api_key),
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=SecretStr(openai_api_key),
)

try:
    loader = TextLoader("data/my_document.txt")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    document_chunks = splitter.split_documents(documents)

    vector_store = Chroma.from_documents(
        document_chunks, embeddings, persist_directory="./chroma_db"
    )
    print("Vector store created successfully")

except Exception as e:
    print(f"Error during ingestion: {e}. Trying to load existing vector store.")
    vector_store = Chroma(
        persist_directory="./chroma_db", embedding_function=embeddings
    )


retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

print("RAG system is ready.")


def get_rag_response(query: str):
    retrieved_docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""
    You are a medical assistant. Use the following information ONLY to answer the question.
    If the answer is not in the context, say 'I cannot find the answer in the provided documents.'

    Context:
    {context}

    Question: {query}
    """

    generated_answer = llm.invoke(prompt)

    print("Running RAGAS validation...")
    data = {
        "question": [query],
        "answer": [generated_answer],
        "contexts": [[doc.page_content for doc in retrieved_docs]],
    }
    dataset = Dataset.from_dict(data)
    result = evaluate(dataset, metrics=[faithfulness])
    faithfulness_score = result["faithfulness"]

    print(f"Faithfulness score: {faithfulness_score}")

    if faithfulness_score[0] < 0.90:
        return "I apologize, but I cannot provide a high-confidence answer based on the retrieved documents. Please consult the original sources."
    else:
        return generated_answer
