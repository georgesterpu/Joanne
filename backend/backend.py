from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
import logging

# Import your existing RAG app functions
from myapp import ask_rag  # Make sure this imports the `ask_rag` function

logging.basicConfig(level=logging.INFO)

app = FastAPI()
templates = Jinja2Templates(directory="templates")  # Create a 'templates' folder

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask")
async def ask_question(data: QueryRequest):
    query = data.query
    logging.info(f"Received query: {query}")
    try:
        response = ask_rag(query)
        return JSONResponse(content={"answer": response})
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return JSONResponse(content={"error": "Failed to process query"}, status_code=500)