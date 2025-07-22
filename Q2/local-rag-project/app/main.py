### Phase 3: The API - Exposing Our Logic to the World

# app/main.py

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from .chains import get_full_chain

# Initialize the FastAPI application
app = FastAPI(
    title="Local RAG API",
    description="An API for a local RAG system with intent detection.",
    version="1.0.0",
)

# Define the data model for the request body
class QueryRequest(BaseModel):
    query: str

async def stream_chain_response(chain, query: str):
    """
    Uses the .astream() method to get a streaming response from the chain.
    """
    async for chunk in chain.astream({"query": query}):
        # Each chunk needs to be in SSE format
        yield f"data: {chunk}\n\n"

@app.post("/ask/stream")
async def ask_question_stream(request: QueryRequest):
    """
    Receives a query, processes it through the RAG pipeline,
    and returns a streaming response.
    """
    chain = get_full_chain(streaming=True)
    return StreamingResponse(
        stream_chain_response(chain, request.query),
        media_type="text/event-stream"
    )

@app.post("/ask")
async def ask_question(request: QueryRequest):
    """
    Non-streaming version of the endpoint for compatibility.
    """
    chain = get_full_chain(streaming=False)
    result = chain.invoke({"query": request.query})
    return {"answer": result}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Local RAG API. Use the /docs endpoint to see the API documentation."}