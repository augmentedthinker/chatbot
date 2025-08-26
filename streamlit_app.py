# Hugging Face Chatbot
# A simple chat interface to interact with Hugging Face language models.
# 
# Instructions:
# 1. Get your Hugging Face API token from https://huggingface.co/settings/tokens
# 2. Enter your token in the sidebar.
# 3. Type your message and press Enter or click 'Send'.
#
# Notes:
# - This demo uses the facebook/blenderbot-400M-distill model via the chat completion endpoint.
# - Conversations are stored in session state and are lost when the app reloads.
# - For best results, be clear and specific in your prompts.

import streamlit as st
import requests

# App title and sidebar setup
st.title("🤖 Hugging Face Chatbot")

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
                "This app uses the Hugging Face Inference API to connect to a language model. "
                "Your conversation history is sent to the model to generate a relevant response.")

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
        # Prepare the messages for the chat model (list of dicts)
        # Hugging Face Inference API chat endpoint expects this format
        hf_messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

        # Call Hugging Face Inference API Chat Completion endpoint
        # Using facebook/blenderbot-400M-distill as it's a known chat model
        API_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill/v1/chat/completions"
        headers = {"Authorization": f"Bearer {st.session_state.hf_token}"}
        payload = {
            "messages": hf_messages,
            "max_tokens": 200, # Limit response length
            "temperature": 0.7 # Add some randomness
        }

        try:
            response = requests.post(API_URL, headers=headers, json=payload)
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

            # Parse the response
            data = response.json()
            # Extract the assistant's reply
            bot_response = data['choices'][0]['message']['content'].strip()

            if not bot_response:
                 bot_response = "..." # Fallback if response is empty

        except requests.exceptions.HTTPError as e:
             if response.status_code == 401:
                 bot_response = "Authentication failed. Please check your Hugging Face API token."
             elif response.status_code == 404:
                 bot_response = "Model not found. The specified model endpoint might be incorrect or unavailable."
             elif response.status_code == 503:
                 bot_response = "Model is currently unavailable or loading. Please try again in a moment."
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
