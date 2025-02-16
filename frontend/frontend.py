import streamlit as st
import requests

st.title("Interactive Q&A with Memory")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Ask a question:")
if st.button("Submit") and query.strip():
    try:
        response = requests.post("http://127.0.0.1:8000/ask", json={"query": query}).json()
        answer = response.get("answer", "No answer received.")

        # Save query & answer in history
        st.session_state.chat_history.append({"question": query, "answer": answer})

        # Display chat history
        for qa in st.session_state.chat_history:
            st.write(f"**You:** {qa['question']}")
            st.write(f"**Bot:** {qa['answer']}")
            st.write("---")

    except requests.exceptions.RequestException as e:
        st.error(f"Error: {e}")