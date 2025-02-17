import streamlit as st
import streamlit.components.v1 as components
import requests

# Configure the page layout (centered) with a professional title
st.set_page_config(page_title="Joanne AI – Market Research Assistant", layout="centered")

st.title("📊 Joanne AI – Your Market Research Assistant")


# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages in reverse order (newest at the bottom)
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user", avatar="👤").write(msg["content"])
    else:
        st.chat_message("assistant", avatar="🤖").write(msg["content"])  # Fixed icon issue

# Function to send query to API
def send_message():
    query = st.session_state.user_input.strip()
    if query:
        # Append user message
        st.session_state.messages.append({"role": "user", "content": query})

        # Display user message
        with st.chat_message("user", avatar="👤"):
            st.write(query)

        # Send request to backend API
        try:
            response = requests.post("http://127.0.0.1:8000/ask", json={"query": query}).json()

            # Handle API quota error
            if "error" in response and "Resource has been exhausted" in response["error"]:
                bot_reply = "⚠️ API quota exceeded. Please try again later."
            else:
                bot_reply = response.get("answer", "No response received.")
                source_info = response.get("source_info", [])

        except requests.exceptions.RequestException:
            bot_reply = "⚠️ Failed to connect to the server. Please try again later."

        # Append bot response to chat history
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})

        # Display bot response as "Joanne AI"
        with st.chat_message("assistant", avatar="🤖"):
            st.write(f"**Joanne AI:** {bot_reply}")
            if source_info:
                with st.expander("View Source Information"):
                    for doc in source_info[:3]:
                        st.markdown(f"**Source:** `{doc['source_file']}`, **Page:** `{doc['page_number']}`")
                        # Embed a PDF viewer for the specific page
                        pdf_filename = doc["source_file"].split("/")[-1]  # Extract filename
                        pdf_url = f"http://127.0.0.1:8000/reports/{pdf_filename}#page={doc['page_number']}"
                        st.markdown(f"[View PDF Page {doc['page_number']}]({pdf_url})")

        # Clear input field
        st.session_state.user_input = ""

# Create a user input box with a Send button
user_input = st.chat_input("Type your question here...")
if user_input:
    st.session_state.user_input = user_input
    send_message()
