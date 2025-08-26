# Hugging Face Chatbot (GPT-2 Fallback)
# A simple text generation interface using the Hugging Face Inference API.
# 
# Instructions:
# 1. Get your Hugging Face API token from https://huggingface.co/settings/tokens
# 2. Enter your token in the sidebar.
# 3. Type your prompt and press Enter or click 'Send'.
#
# Notes:
# - This demo uses the 'gpt2' model via the general inference endpoint.
# - Responses are generated based on the prompt history.
# - Conversations are stored in session state and are lost when the app reloads.

import streamlit as st
import requests

# App title and sidebar setup
st.title("🤖 Hugging Face Text Gen (GPT-2)")

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
        "3. Type your prompt and press Enter."
    )
    hf_token = st.text_input("Hugging Face API Token:", type="password", key="token_input")
    if st.button("Save Token"):
        st.session_state.hf_token = hf_token
        st.success("Token saved!")

    st.markdown("---")
    st.markdown("**How it works:**\n"
                "This app uses the Hugging Face Inference API with the 'gpt2' model. "
                "Your prompt history is formatted and sent to the model for text generation.")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Truncate long messages for display in chat history
        display_content = (message["content"][:100] + '...') if len(message["content"]) > 100 else message["content"]
        st.markdown(display_content)

# React to user input
if prompt := st.chat_input("Enter a prompt..."):
    # Add user message to history
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
        # Simple concatenation of the last few user messages for context
        # GPT-2 might not handle complex multi-turn chat as well, so keep it simple
        recent_history = st.session_state.messages[-4:] # Get last 4 messages for context
        formatted_prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])
        formatted_prompt += "\nassistant:"

        # Call Hugging Face Inference API - General Endpoint for gpt2
        API_URL = "https://api-inference.huggingface.co/models/gpt2"
        headers = {"Authorization": f"Bearer {st.session_state.hf_token}"}
        payload = {
            "inputs": formatted_prompt,
            "parameters": {
                "max_new_tokens": 80, # Limit response length
                "return_full_text": False, # Only get the new text
                "temperature": 0.7, # Add some randomness
                "top_k": 50, # Limit to top 50 tokens for stability
                "stop": ["\nuser:", "\nassistant:"] # Try to stop generation at next prompt part
            }
        }

        try:
            response = requests.post(API_URL, headers=headers, json=payload)
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

            # Parse the response
            data = response.json()
            # With return_full_text=False, we expect the new text directly
            if isinstance(data, list) and len(data) > 0:
                bot_response_raw = data[0].get('generated_text', '').strip()
            elif isinstance(data, dict):
                bot_response_raw = data.get('generated_text', '').strip()
            else:
                bot_response_raw = ""

            # Basic cleaning: stop at specific markers if they appear in the raw response
            bot_response = bot_response_raw
            for stop_marker in ["\nuser:", "\nassistant:"]:
                if stop_marker in bot_response:
                    bot_response = bot_response.split(stop_marker)[0].strip()

            if not bot_response:
                 bot_response = "..." # Fallback if response is empty or parsing failed

        except requests.exceptions.HTTPError as e:
             if response.status_code == 401:
                 bot_response = "Authentication failed. Please check your Hugging Face API token."
             elif response.status_code == 404:
                 bot_response = "Model 'gpt2' not found via this endpoint. Please check the model name."
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
            bot_response = f"Sorry, I received an unexpected response from the AI service. Please try again later. (Error details: {e})"

        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(bot_response)
