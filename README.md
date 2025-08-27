# Tiny Chat MVP (Streamlit)

Minimal chat UI. Works without keys (Demo mode) and can switch to Hugging Face Inference API with a token.

## Steps
1. Create a public GitHub repo with these files.
2. Deploy on Streamlit Community Cloud (`streamlit_app.py` as main file).
3. Add a Secret in Streamlit:
```
HF_TOKEN = your_hf_api_token_here
```
4. Open your app URL.

Demo mode works instantly. Hugging Face mode uses your token for real responses.
