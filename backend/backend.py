from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
import logging
from myapp import ask_rag  # Import the updated `ask_rag` function

logging.basicConfig(level=logging.INFO)

app = FastAPI()
templates = Jinja2Templates(directory="templates")  # Ensure a 'templates' folder exists

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask")
async def ask_question(data: QueryRequest):
    """Handles incoming queries from the frontend and returns AI responses."""
    query = data.query
    logging.info(f"Received query: {query}")
    try:
        response = ask_rag(query)
        return JSONResponse(content=response)  # Return full JSON response
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return JSONResponse(content={"error": "Failed to process query"}, status_code=500)
