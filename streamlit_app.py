# streamlit_app.py (Token Tester)

import streamlit as st
import requests

# --- Page Configuration ---
st.set_page_config(
    page_title="HF Token Tester",
    page_icon="🔑",
    layout="centered"
)

st.title("🔑 Hugging Face API Token Tester")
st.markdown(
    """
    This tool checks if your Hugging Face API token is valid. We've been getting `404` errors, 
    so let's confirm your token is working before we continue debugging the chatbot. A valid 
    token should return a `200 OK` status.
    """
)

# --- API Configuration ---
# This is a special endpoint just for checking authentication
WHOAMI_URL = "https://api-inference.huggingface.co/whoami"

# --- User Input ---
st.header("Enter Your Token to Test")
hf_api_token = st.text_input(
    "Hugging Face API Token", type="password", help="Your token is not stored."
)

if st.button("Test Token"):
    if not hf_api_token:
        st.warning("Please enter your API token.")
    else:
        with st.spinner("Checking token..."):
            headers = {"Authorization": f"Bearer {hf_api_token}"}
            
            try:
                response = requests.get(WHOAMI_URL, headers=headers, timeout=10)
                
                st.subheader("Test Result")
                
                # Display the raw status code for clarity
                st.write(f"**HTTP Status Code:** `{response.status_code}`")

                # Interpret the result for the user
                if response.status_code == 200:
                    st.success("✅ Your token is valid!")
                    st.markdown("The API recognized you. Here is the information it returned:")
                    st.json(response.json())
                    st.info("Now that we know the token works, we can go back to the chatbot code. The issue is definitely with the availability of the models on the free tier.")
                
                elif response.status_code == 401:
                    st.error("❌ Your token is invalid or incorrect.")
                    st.markdown("The API returned a `401 Unauthorized` error. Please double-check your token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).")
                    st.json(response.json())
                
                else:
                    st.error(f"🚨 An unexpected error occurred.")
                    st.markdown(f"The API returned a status code of **{response.status_code}**. This is not a token error, but something else. Here is the full response:")
                    st.json(response.json())

            except requests.exceptions.RequestException as e:
                st.error(f"A network error occurred: {e}")
