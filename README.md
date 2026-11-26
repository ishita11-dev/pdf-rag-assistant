# 📄 PDF AI Assistant (RAG Pipeline)

An interactive Retrieval-Augmented Generation (RAG) web application that allows users to upload any PDF document and ask natural-language questions grounded strictly in the document's content.

## 🛠️ Tech Stack
- **Language:** Python 3.11
- **Frontend UI:** Streamlit
- **Orchestration & Chunking:** LangChain (`RecursiveCharacterTextSplitter`)
- **Vector Database:** FAISS (Local Similarity Search)
- **LLM & Embeddings:** Google Gemini (`gemini-3.5-flash-lite` & `gemini-embedding-001`)

## ⚙️ How It Works
1. **Extraction & Chunking:** Extracts raw text via `pypdf` and splits it into 1,000-character overlapping chunks.
2. **Vector Indexing:** Generates semantic embeddings using Gemini and stores them in an in-memory FAISS index.
3. **Grounded Q&A:** Retrieves the top 3 most relevant chunks per query and prompts Gemini to answer strictly from that context to prevent hallucinations.

## 🚀 Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
