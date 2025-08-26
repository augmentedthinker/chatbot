# Hugging Face Chatbot (DialoGPT Fallback)
# A simple chat interface to interact with Hugging Face language models.
# 
# Instructions:
# 1. Get your Hugging Face API token from https://huggingface.co/settings/tokens
# 2. Enter your token in the sidebar.
# 3. Type your message and press Enter or click 'Send'.
#
# Notes:
# - This demo uses the microsoft/DialoGPT-medium model via the general inference endpoint.
# - Conversations are stored in session state and are lost when the app reloads.
# - For best results, be clear and specific in your prompts.

import streamlit as st
import requests

# App title and sidebar setup
st.title("🤖 Hugging Face Chatbot (DialoGPT)")

# Initialize session state for messages and API token
if "messages" not in st.session_state:
    st.session_state.messages = []
if "hf_token" not in st.session_state:
    st.session_state.hf_token = ""

# Sidebar for instructions and API token input
with st.sidebar:
    st.header("Setup")
    st.markdown(
        "1. Get your Hugging Face API token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)\n"
        "2. Paste it below.\n"
        "3. Type your message and press Enter."
    )
    hf_token = st.text_input("Hugging Face API Token:", type="password", key="token_input")
    if st.button("Save Token"):
        st.session_state.hf_token = hf_token
        st.success("Token saved!")

    st.markdown("---")
    st.markdown("**How it works:**\n"
                "This app uses the Hugging Face Inference API with the microsoft/DialoGPT-medium model. "
                "Your conversation history is formatted and sent as a prompt to the model.")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What would you like to say?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check if API token is available
    if not st.session_state.hf_token:
        with st.chat_message("assistant"):
            st.error("Please enter your Hugging Face API token in the sidebar.")
    else:
        # Prepare the prompt for the model
        # Format conversation history for DialoGPT
        formatted_prompt = ""
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                formatted_prompt += f"You: {msg['content']}\n"
            elif msg["role"] == "assistant":
                formatted_prompt += f"Chatbot: {msg['content']}\n"
        # Add the prompt for the next assistant response
        formatted_prompt += "Chatbot:"

        # Call Hugging Face Inference API - General Endpoint for DialoGPT
        API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
        headers = {"Authorization": f"Bearer {st.session_state.hf_token}"}
        payload = {
            "inputs": formatted_prompt,
            "parameters": {
                "max_new_tokens": 100,
                "return_full_text": False, # Important: Only get the new text
                "temperature": 0.8
            }
        }

        try:
            response = requests.post(API_URL, headers=headers, json=payload)
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

            # Parse the response
            data = response.json()
            # With return_full_text=False, we expect the new text directly
            # Sometimes it's a list with one dict, sometimes just the dict
            if isinstance(data, list) and len(data) > 0:
                bot_response_raw = data[0].get('generated_text', '').strip()
            elif isinstance(data, dict):
                bot_response_raw = data.get('generated_text', '').strip()
            else:
                bot_response_raw = ""

            # Simple cleaning: stop at the next "You:" prompt if generated
            if "\nYou:" in bot_response_raw:
                bot_response = bot_response_raw.split("\nYou:")[0].strip()
            else:
                bot_response = bot_response_raw

            if not bot_response:
                 bot_response = "..." # Fallback if response is empty or parsing failed

        except requests.exceptions.HTTPError as e:
             if response.status_code == 401:
                 bot_response = "Authentication failed. Please check your Hugging Face API token."
             elif response.status_code == 404:
                 bot_response = "Model microsoft/DialoGPT-medium not found via this endpoint. Please check the model name or try another."
             elif response.status_code == 503:
                 # Common for HF Inference API if model needs to load
                 error_msg = response.json().get('error', 'Model loading or unavailable.')
                 bot_response = f"Service temporarily unavailable: {error_msg}"
             else:
                 bot_response = f"HTTP error occurred: {e}"
        except requests.exceptions.RequestException as e:
            # Handle other network errors
            bot_response = f"Sorry, I encountered a network error contacting the AI service: {e}"
        except (KeyError, IndexError, TypeError) as e:
            # Handle unexpected response format
            bot_response = f"Sorry, I received an unexpected response from the AI service. It might be busy, unavailable, or the response format changed. Please try again later. (Error details: {e})"

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(bot_response)
