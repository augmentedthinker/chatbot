# streamlit_app.py

import streamlit as st
import requests
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="Vibe Coder Chatbot",
    page_icon="🤖",
    layout="centered"
)

# --- Application Title and Description ---
st.title("🤖 Vibe Coder Chatbot")
st.markdown("A minimal, deployable chatbot using the Hugging Face Inference API. Built for the 'Vibe Coder' workflow.")

# --- Sidebar for Instructions and Configuration ---
with st.sidebar:
    st.header("Configuration")
    st.markdown(
        """
        1.  Get your Hugging Face API token from [here](https://huggingface.co/settings/tokens).
        2.  Enter it below.
        3.  Choose a model to chat with.
        """
    )
    # Get user's Hugging Face API token
    hf_api_token = st.text_input(
        "Hugging Face API Token", type="password", help="Your token is not stored."
    )
    
    # NEW: Allow user to select the model
    # Changed default to a more reliable, conversational model
    model_name = st.text_input(
        "Hugging Face Model ID",
        "google/gemma-2b-it",
        help="e.g., 'distilgpt2' or 'microsoft/DialoGPT-medium'"
    )

    st.markdown("---")
    st.info(f"Currently chatting with: **{model_name}**")

# --- Chat History Management ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Please enter your Hugging Face API token to begin."}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- API Query Function ---
def query_hf_api(prompt, token, model):
    """
    Sends a prompt to the Hugging Face Inference API and returns the response.
    """
    if not token:
        return {"error": "Hugging Face API token is missing. Please enter it in the sidebar."}

    # The API URL is now dynamic based on the selected model
    api_url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "return_full_text": False,
            "max_new_tokens": 250,
            "temperature": 0.7,
            "top_p": 0.9,
        },
        "options": {"wait_for_model": True}
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        # Improved error message for 404
        if response.status_code == 404:
            return {"error": f"Model '{model}' not found or not available on the free Inference API. Please try another model."}
        if response.status_code == 401:
            return {"error": "Authentication failed. Please check your API token."}
        if response.status_code == 503:
             return {"error": "The model is loading. Please try again in a few moments."}
        return {"error": f"An HTTP error occurred: {http_err}"}
    except requests.exceptions.RequestException as req_err:
        return {"error": f"A network error occurred: {req_err}"}

# --- Main Chat Logic ---
if user_prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")

        # IMPROVED: Construct a prompt suitable for chat models like Gemma
        # This helps the model understand the conversation flow
        prompt_history = ""
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                prompt_history += f"<start_of_turn>user\n{msg['content']}<end_of_turn>\n"
            else:
                # We skip adding previous assistant messages to keep the prompt focused
                # on the user's direct lineage of questions.
                pass
        
        # Add the final instruction for the model to generate its response
        final_prompt = prompt_history + "<start_of_turn>model\n"

        # Query the API with the selected model
        api_response = query_hf_api(final_prompt, hf_api_token, model_name)

        if "error" in api_response:
            bot_response = api_response["error"]
            st.error(bot_response) # Display error in the chat
        elif isinstance(api_response, list) and api_response and "generated_text" in api_response[0]:
            bot_response = api_response[0]["generated_text"].strip()
        else:
            bot_response = "Sorry, I received an unexpected response from the API."
            st.warning(f"Unexpected API response format: {api_response}")

        message_placeholder.markdown(bot_response)

    st.session_state.messages.append({"role": "assistant", "content": bot_response})
