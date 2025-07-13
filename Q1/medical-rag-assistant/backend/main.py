# main.py
from fastapi import FastAPI
from endpoints import router

# Create the FastAPI app instance
app = FastAPI(
    title="Medical AI Assistant",
    description="A production-ready RAG system for medical queries.",
    version="1.0.0",
)

# Include the router from endpoints.py
# All routes defined in that file will be added to our app.
app.include_router(router)


@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Medical AI Assistant API. Go to /docs for more info."
    }
