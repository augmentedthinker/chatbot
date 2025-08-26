# Hugging Face Chatbot
# A simple chat interface to interact with Hugging Face language models.
# 
# Instructions:
# 1. Get your Hugging Face API token from https://huggingface.co/settings/tokens
# 2. Enter your token in the sidebar.
# 3. Type your message and press Enter or click 'Send'.
#
# Notes:
# - This demo uses the microsoft/DialoGPT-medium model.
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
        # Prepare the prompt for the model
        # Format conversation history for DialoGPT
        formatted_prompt = ""
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                formatted_prompt += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                formatted_prompt += f"Assistant: {msg['content']}\n"
        # Add the prompt for the next assistant response
        formatted_prompt += "Assistant:"

        # Call Hugging Face Inference API
        API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
        headers = {"Authorization": f"Bearer {st.session_state.hf_token}"}
        payload = {
            "inputs": formatted_prompt,
            # "parameters": {"max_new_tokens": 100, "return_full_text": False} # Optional parameters
        }

        try:
            response = requests.post(API_URL, headers=headers, json=payload)
            response.raise_for_status()  # Raise an exception for bad status codes

            # Parse the response
            data = response.json()
            # DialoGPT often returns the full text. We need to extract the new part.
            # The model might append to the prompt, so we find the part after our prompt.
            full_text = data[0]['generated_text']
            if full_text.startswith(formatted_prompt):
                bot_response = full_text[len(formatted_prompt):].strip()
            else:
                # Fallback if structure is unexpected
                bot_response = full_text.strip()

            # Ensure we only get the first response part if multiple turns are generated
            # This is a simple heuristic, might need refinement for other models
            if "\nUser:" in bot_response:
                bot_response = bot_response.split("\nUser:")[0].strip()

            if not bot_response:
                 bot_response = "..." # Fallback if parsing fails

        except requests.exceptions.RequestException as e:
            # Handle API call errors
            bot_response = f"Sorry, I encountered an error contacting the AI service: {e}"
        except (KeyError, IndexError, TypeError) as e:
            # Handle unexpected response format
            bot_response = f"Sorry, I received an unexpected response from the AI service. It might be busy or unavailable. Please try again later. (Error: {e})"

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(bot_response)
