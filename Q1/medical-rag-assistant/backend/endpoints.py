from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from rag import get_rag_response

router = APIRouter()


class QueryRequest(BaseModel):
    query: str


@router.post("/query")
async def query_rag_system(request: QueryRequest):
    try:
        response = get_rag_response(request.query)
        return {"query": request.query, "response": response}
    except Exception as e:
        # This provides better error information if something goes wrong.
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
