import streamlit as st
import requests

st.title("RAG-Powered Q&A")

query = st.text_input("Ask a question:")
if st.button("Submit") and query.strip():
    try:
        response = requests.post("http://127.0.0.1:8000/ask", json={"query": query})

        # Extract and display answer
        data = response.json()
        answer = data.get("answer")  # Ensure "answer" key exists
        if answer:
            st.write("### Answer:")
            st.write(answer)
        else:
            st.warning("No answer received from API.")

    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to FastAPI: {e}")