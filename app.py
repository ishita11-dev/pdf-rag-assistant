import os
import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Page setup
st.set_page_config(page_title="PDF AI Assistant", page_icon="📄")
st.title("📄 Chat With Your PDF")

# Sidebar for API Key & File Upload
with st.sidebar:
    api_key = st.text_input("Paste your Gemini API Key:", type="password")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Helper function to read and index the PDF
def build_search_index(pdf_file, key):
    reader = PdfReader(pdf_file)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() or ""

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_text(full_text)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=key
    )
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_store

# Main App Logic
if api_key and uploaded_file:
    if "vector_store" not in st.session_state:
        with st.spinner("Reading and indexing your PDF..."):
            st.session_state.vector_store = build_search_index(uploaded_file, api_key)
        st.success("PDF ready! Ask a question below.")

    user_question = st.text_input("Ask something about your PDF:")

    if user_question:
        with st.spinner("Searching PDF and thinking..."):
            docs = st.session_state.vector_store.similarity_search(user_question, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])

            llm = ChatGoogleGenerativeAI(
                model="gemini-3.5-flash-lite",
                google_api_key=api_key,
                temperature=0.2
            )
            prompt = f"""Answer the question using ONLY the context below. 
If the answer is not in the context, say "I couldn't find that in the document."

Context:
{context}

Question: {user_question}"""

            response = llm.invoke(prompt)

            st.write("### Answer:")
            if isinstance(response.content, list):
                clean_answer = "".join(
                    block.get("text", "") for block in response.content if isinstance(block, dict)
                )
                st.write(clean_answer)
            else:
                st.write(response.content)
else:
    st.info("👈 Enter your Gemini API key and upload a PDF in the left sidebar to begin.")