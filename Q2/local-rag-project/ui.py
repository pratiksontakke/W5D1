# ui.py
import streamlit as st
import requests
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="SaaS Support Bot",
    page_icon="🤖"
)

st.title("🤖 SaaS Support Bot")
st.caption("Your friendly AI assistant powered by local LLMs.")

# --- API Endpoints ---
API_BASE_URL = "http://127.0.0.1:8000"
STREAM_API_URL = f"{API_BASE_URL}/ask/stream"
REGULAR_API_URL = f"{API_BASE_URL}/ask"

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask me anything about our service..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Use the streaming endpoint
            with requests.post(STREAM_API_URL, json={"query": prompt}, stream=True) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        # Remove the "data: " prefix and decode
                        line_text = line.decode('utf-8').replace('data: ', '')
                        full_response += line_text
                        # Add a blinking cursor to show it's still thinking
                        message_placeholder.markdown(full_response + "▌")
            
            # Final update without the cursor
            message_placeholder.markdown(full_response)
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Error: {str(e)}"
            message_placeholder.error(error_msg)
            full_response = error_msg

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})