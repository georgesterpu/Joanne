import os
import logging
from dotenv import load_dotenv
from spellchecker import SpellChecker
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
import google.generativeai as genai

spell = SpellChecker()

def preprocess_query(query):
    """Preprocess the query for better understanding."""
    corrected_query = " ".join([spell.correction(word) for word in query.split()])
    return corrected_query

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load API key from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Load and process the document
files = ['../reports/2024-conocophillips-proxy-statement.pdf',
          '../reports/2023-conocophillips-aim-presentation-1.pdf'] # OCR'd offline
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
# Initialize LLM
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp', google_api_key=GEMINI_API_KEY)

# Create RAG pipeline
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
prompt = ChatPromptTemplate.from_template(
    """
    You are a fun, bubbly, and helpful AI assistant named Joanne.
    Your role is to provide brilliant insights and answers to any questions asked by the user.
    Whatever you do, make sure to stand out with each response to impress and inspire the user.
    When it comes to less obvious questions, challenge the user to think critically, instead of just giving them the answer,
    so they benefit from the interaction with you.
    Use the following pieces of context and chat history to answer the user's question.
    --------------------
    Chat History: {chat_history}
    --------------------
    Context: {context}
    --------------------
    Question: {input}
    """
)

document_chain = create_stuff_documents_chain(llm, prompt)
qa_chain = create_retrieval_chain(retriever, document_chain)

def ask_rag(query):
    """Handles queries while maintaining conversation history."""
    # query = preprocess_query(query)  # possibly buggy, leaving out for now
    
    # Retrieve stored chat history
    chat_history = memory.load_memory_variables({})["chat_history"]

    # Add user's query to history
    chat_history.append(HumanMessage(content=query))

    # Prepare the input for the RAG chain
    input_data = {
        "input": query,
        "chat_history": chat_history
    }
    
    response = qa_chain.invoke(input_data)
    
    # Extract answer and source documents
    answer = response.get("answer") if isinstance(response, dict) else "No relevant information found."
    source_documents = response.get("context", [])

    # Add AI's response to chat history
    chat_history.append(AIMessage(content=answer))

    # Update stored memory with new chat history
    memory.save_context({"input": query}, {"output": answer})

    # Extract source information (file and page number) from source documents
    source_info = []
    for doc in source_documents:
        source_info.append({
            "source_file": doc.metadata.get("source", "Unknown"),
            "page_number": doc.metadata.get("page", 0),
            "content": doc.page_content  # Optional: include the actual content
        })

    # Convert chat history to JSON-serializable format
    formatted_history = [
        {"role": "user" if isinstance(msg, HumanMessage) else "bot", "content": msg.content}
        for msg in chat_history
    ]

    # Return structured response with source data
    return {
        "answer": answer,
        "source_info": source_info,  # Include source data
        "chat_history": formatted_history  # JSON-serializable
    }
