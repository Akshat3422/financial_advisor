import streamlit as st
import requests

BASE_URL = "http://127.0.0.1:8000"

st.title("Document RAG System")

# ----------------- Fetch Document from URL -----------------
st.header("1️⃣ Fetch Document from URL")
url = st.text_input("Enter the document URL:")
if st.button("Fetch Document"):
    if url:
        with st.spinner("Downloading document..."):
            response = requests.post(f"{BASE_URL}/fetch_document/", params={"url": url})
            if response.status_code == 200:
                st.success(f"File saved and indexed successfully!")
                st.json(response.json())
            else:
                st.error(f"Error: {response.json().get('detail', 'Unknown error')}")

# ----------------- Upload Document -----------------
st.header("2️⃣ Upload Document")
uploaded_file = st.file_uploader("Choose a file to upload", type=["pdf", "docx"])
if st.button("Upload Document"):
    if uploaded_file is not None:
        with st.spinner("Uploading document..."):
            files = {"upload_file": (uploaded_file.name, uploaded_file.getvalue())}
            response = requests.post(f"{BASE_URL}/upload_document/", files=files)
            if response.status_code == 200:
                st.success(f"File uploaded and indexed successfully!")
                st.json(response.json())
            else:
                st.error(f"Error: {response.json().get('detail', 'Unknown error')}")

# ----------------- Submit a Question -----------------
st.header("3️⃣ Ask a Question")
question = st.text_input("Enter your question here:")
if st.button("Submit Question"):
    if question:
        with st.spinner("Querying documents..."):
            response = requests.get(f"{BASE_URL}/submit/", params={"question": question})
            if response.status_code == 200:
                st.success("Answer:")
                st.write(response.json().get("result"))
            else:
                st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
