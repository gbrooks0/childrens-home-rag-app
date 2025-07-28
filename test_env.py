# test_env.py
import streamlit as st
import os

st.set_page_config(page_title="Environment Test", layout="centered")
st.title("Environment Test App")

st.write("---")
st.header("1. Basic App Check")
st.success("Hello, Streamlit! This app is running.")

st.write("---")
st.header("2. API Key Environment Variable Check")

try:
    google_key = os.environ.get("GOOGLE_API_KEY")
    if google_key:
        st.write(f"GOOGLE_API_KEY: Found (starts with '{google_key[:5]}...')")
    else:
        st.error("GOOGLE_API_KEY: NOT FOUND in os.environ!")

    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        st.write(f"OPENAI_API_KEY: Found (starts with '{openai_key[:5]}...')")
    else:
        st.error("OPENAI_API_KEY: NOT FOUND in os.environ!")

except Exception as e:
    st.exception(e)
    st.error(f"Error checking API keys: {e}")

st.write("---")
st.header("3. Python Version Check")
st.write(f"Python Version: {sys.version}")

st.write("---")
st.header("4. System Path Check")
# This might be long, but useful if there's a module loading issue
# st.write("sys.path:")
# for p in sys.path:
#     st.write(f"- {p}")
