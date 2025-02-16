import os
import logging
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import google.generativeai as genai
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load API key from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Load and process the document
files = ['reports/2024-conocophillips-proxy-statement.pdf',
          'reports/2023-conocophillips-aim-presentation-1.pdf'] # OCR'd offline
# Create vector store using FAISS
embedding_model = HuggingFaceEmbeddings()

faiss_index_path = "faiss_index"
# Check if FAISS index already exists
if os.path.exists(faiss_index_path + "/index.faiss"):
    logging.info("Loading existing FAISS index...")
    vector_store = FAISS.load_local(faiss_index_path, embedding_model, allow_dangerous_deserialization=True)
else:
    logging.info("Creating new FAISS index...")
    raw_docs = sum([PyMuPDFLoader(f).load() for f in files], [])
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_docs)
    vector_store = FAISS.from_documents(documents, embedding_model)
    vector_store.save_local(faiss_index_path)  # Save FAISS index

# Create retriever
retriever = vector_store.as_retriever()
# Initialize LLMM
llm = ChatGoogleGenerativeAI(model='gemini-2.0-pro-exp-02-05', google_api_key=GEMINI_API_KEY)

## Create RAG pipeline
# Add conversational memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# Conversational Retrieval Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm, retriever=retriever, memory=memory)

def ask_rag(query):
    """Handles queries while maintaining conversation history."""
    
    # Retrieve stored chat history
    chat_history = memory.load_memory_variables({})["chat_history"]

    # Add user's query to history
    chat_history.append(HumanMessage(content=query))

    # Invoke RAG chain
    response = qa_chain.invoke({"question": query, "chat_history": chat_history})
    
    # Extract answer
    answer = response.get("answer") if isinstance(response, dict) else "No relevant information found."

    # Add AI's response to chat history
    chat_history.append(AIMessage(content=answer))

    # Update stored memory with new chat history
    memory.save_context({"input": query}, {"output": answer})

    # Convert chat history to JSON-serializable format
    formatted_history = [
        {"role": "user" if isinstance(msg, HumanMessage) else "bot", "content": msg.content}
        for msg in chat_history
    ]

    # Return structured response
    return {
        "answer": answer,
        "chat_history": formatted_history  # JSON-serializable
    }

# Run test query (Optional)
if __name__ == "__main__":
    print(ask_rag("What is the purpose of the proxy statement?"))
