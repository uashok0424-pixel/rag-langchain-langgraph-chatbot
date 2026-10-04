\# ☕ Java RAG Chatbot



An AI-powered Java study assistant built using Retrieval-Augmented Generation (RAG).



The chatbot answers questions from Java study notes using:



\- Python

\- Streamlit

\- LangChain

\- LangGraph

\- Ollama

\- FAISS

\- Llama 3.2



\## 🚀 Features



\- Ask questions about Java concepts

\- Retrieves relevant information from Java notes

\- Uses FAISS for vector similarity search

\- Uses Ollama for local LLM responses

\- LangGraph manages the RAG workflow

\- Streamlit provides the web interface

\- Shows retrieved sources for answers

\- Prevents the model from inventing answers when information is unavailable



\## 🏗️ Architecture



User Question

&#x20;     ↓

Streamlit Web App

&#x20;     ↓

LangGraph

&#x20;     ↓

FAISS Vector Store

&#x20;     ↓

Relevant Java Notes

&#x20;     ↓

Ollama / Llama 3.2

&#x20;     ↓

Generated Answer

&#x20;     ↓

Streamlit UI



\## 📁 Project Structure



```text

rag-langchain-langgraph-chatbot/

│

├── app/

│   ├── graph.py

│   ├── rag.py

│   └── web\_app.py

│

├── data/

│   └── documents/

│       └── java\_notes.pdf

│

├── src/

│   ├── \_\_init\_\_.py

│   ├── config.py

│   └── ingest.py

│

├── vectorstore/

│   └── faiss\_index/

│       ├── index.faiss

│       └── index.pkl

│

├── .gitignore

└── README.md

