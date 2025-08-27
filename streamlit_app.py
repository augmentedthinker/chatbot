# streamlit_app.py
# Minimal Chat MVP for Streamlit Community Cloud
# - Demo mode: no API keys; local rule-based replies so the UI always works.
# - Real mode: use Hugging Face Inference API (HF Router) if HF_TOKEN is set.

import os
import time
import requests
import streamlit as st

APP_TITLE = "Tiny Chat MVP"

st.set_page_config(page_title=APP_TITLE, page_icon="💬", layout="centered")
st.title("💬 Tiny Chat MVP")

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Mode", ["Demo (no key)", "Hugging Face Inference API"], index=0)
    model = st.selectbox(
        "Model (HF Router id)",
        ["openai/gpt-oss-20b", "meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-7B-Instruct"],
        index=0,
    )
    system_prompt = st.text_area("System prompt", "You are a concise, friendly assistant.")
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)

    token_from_secrets = st.secrets.get("HF_TOKEN", None)
    token_from_env = os.environ.get("HF_TOKEN")
    token_manual = ""
    if mode == "Hugging Face Inference API":
        token_manual = st.text_input("HF_TOKEN (optional if set in Secrets)", type="password")
    resolved_token = token_from_secrets or token_from_env or token_manual

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": system_prompt}]

if st.session_state.messages and st.session_state.messages[0]["role"] == "system":
    st.session_state.messages[0]["content"] = system_prompt

def demo_reply(user_text: str) -> str:
    canned = {"hello": "Hey! 👋", "hi": "Hi there!", "help": "Tell me your goal.", "joke": "Cache joke."}
    low = user_text.strip().lower()
    for k, v in canned.items():
        if k in low:
            return v
    return f"I hear you said: “{user_text}”. Next action: clarify your goal."

def hf_chat_completion(messages, model_id: str, temperature: float, token: str):
    api_url = "https://router.huggingface.co/v1/chat/completions"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"model": model_id, "messages": messages, "temperature": temperature, "stream": False}
    r = requests.post(api_url, headers=headers, json=payload, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"HF API error {r.status_code}: {r.text[:300]}")
    data = r.json()
    return data["choices"][0]["message"]["content"]

for m in st.session_state.messages[1:]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_input = st.chat_input("Type a message…")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            if mode == "Demo (no key)":
                for dots in ["", ".", "..", "..."]:
                    placeholder.markdown(f"_thinking{dots}_")
                    time.sleep(0.15)
                reply = demo_reply(user_input)
            else:
                if not resolved_token:
                    raise RuntimeError("No HF_TOKEN found.")
                api_messages = st.session_state.messages
                reply = hf_chat_completion(api_messages, model, temperature, resolved_token)
            placeholder.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            placeholder.markdown(f"⚠️ {e}")

with st.sidebar:
    if st.button("Clear chat"):
        st.session_state.messages = [{"role": "system", "content": system_prompt}]
        st.experimental_rerun()
