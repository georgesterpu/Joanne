import os
import logging
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load API key from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Load and process the document
loader = TextLoader("sample_report.txt")
docs = loader.load()

# Split document into chunks for embedding
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
documents = text_splitter.split_documents(docs)

# Create vector store using FAISS
embedding_model = HuggingFaceEmbeddings()

faiss_index_path = "faiss_index"
# Check if FAISS index already exists
if os.path.exists(faiss_index_path + "/index.faiss"):
    logging.info("Loading existing FAISS index...")
    vector_store = FAISS.load_local(faiss_index_path, embedding_model, allow_dangerous_deserialization=True)
else:
    logging.info("Creating new FAISS index...")
    vector_store = FAISS.from_documents(documents, embedding_model)
    vector_store.save_local(faiss_index_path)  # Save FAISS index
vector_store = FAISS.from_documents(documents, embedding_model)

# Create retriever
retriever = vector_store.as_retriever()

# Initialize LLMM
llm = ChatGoogleGenerativeAI(model='gemini-2.0-pro-exp-02-05', google_api_key=GEMINI_API_KEY)

# Create RAG pipeline
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

# Function to query the RAG system
def ask_rag(query):
    print(f"\n Query: {query}")
    response = qa_chain.invoke(query)
    print(f"Answer: {response}")

# Run some test queries
if __name__ == "__main__":
    ask_rag("What are the major trends in AI?")
    ask_rag("Which companies are leading AI research?")
    ask_rag("What are the biggest AI investment areas?")
