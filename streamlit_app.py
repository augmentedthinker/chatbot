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

# --- Hugging Face API Configuration ---
MODEL_NAME = "gpt2"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"

# --- Sidebar for Instructions and API Token ---
with st.sidebar:
    st.header("Instructions")
    st.markdown(
        """
        1.  Get your Hugging Face API token from [here](https://huggingface.co/settings/tokens).
        2.  Enter your token below to activate the chatbot.
        3.  Start chatting! The conversation history is used to provide context.
        """
    )
    # Securely get API token from user
    hf_api_token = st.text_input("Hugging Face API Token", type="password", help="Your token is not stored.")
    st.markdown("---")
    st.info("This app uses the `gpt2` model. It's a foundational model, so responses may be simple or repetitive.")

# --- Chat History Management ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Please enter your Hugging Face API token to begin."}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- API Query Function ---
def query_hf_api(prompt, token):
    """
    Sends a prompt to the Hugging Face Inference API and returns the response.
    Handles potential errors gracefully.
    """
    if not token:
        return {"error": "Hugging Face API token is missing. Please enter it in the sidebar."}

    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "return_full_text": False, # Only return the generated text
            "max_new_tokens": 150,     # Limit the length of the response
            "temperature": 0.7,        # Adjust creativity
            "top_p": 0.9,              # Nucleus sampling
        },
        "options": {
            "wait_for_model": True # Avoid 503 errors if the model is loading
        }
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        # Specifically handle common HTTP errors with user-friendly messages
        if response.status_code == 401:
            return {"error": "Authentication failed. Please check if your API token is correct."}
        elif response.status_code == 429:
            return {"error": "Too many requests. Please wait a bit before trying again."}
        elif response.status_code == 503:
             return {"error": "The model is currently loading or unavailable. Please try again in a few moments."}
        else:
            return {"error": f"An HTTP error occurred: {http_err}"}
    except requests.exceptions.RequestException as req_err:
        return {"error": f"A network error occurred: {req_err}"}
    except json.JSONDecodeError:
        return {"error": "Failed to decode the response from the API. The API might be temporarily down."}


# --- Main Chat Logic ---
if user_prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")

        # Construct a single prompt from the conversation history
        # Simple approach: join all messages
        conversation_history = "\n".join([msg["content"] for msg in st.session_state.messages])
        
        # Query the API
        api_response = query_hf_api(conversation_history, hf_api_token)

        if "error" in api_response:
            bot_response = api_response["error"]
            st.error(bot_response) # Show error in the UI
        elif isinstance(api_response, list) and api_response and "generated_text" in api_response[0]:
            bot_response = api_response[0]["generated_text"].strip()
        else:
            bot_response = "Sorry, I received an unexpected response from the API. Please try again."
            st.warning(f"Unexpected API response format: {api_response}")

        message_placeholder.markdown(bot_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
