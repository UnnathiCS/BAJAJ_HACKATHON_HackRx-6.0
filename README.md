# BAJAJ_HACKATHON_HackRx-6.0
# 🛡️ HackRx Insurance Policy Q&A API

This is a FastAPI-based insurance policy question-answering API, built for the **HackRx 6.0 hackathon**. The API accepts a PDF of an insurance policy and a list of questions, then uses semantic search to find the most relevant clauses from the document and return answers.

---

## Features

- ✅ Upload PDF via URL
- ✅ Ask natural-language questions about the policy
- ✅ Uses semantic similarity (MiniLM model)
- ✅ Extracts clauses using PyMuPDF
- ✅ Works within HackRx webhook format
- ✅ Lightweight, deployable on Railway (or any cloud)

---

## Tech Stack

- **FastAPI** – REST API framework
- **Uvicorn** – ASGI server
- **Sentence Transformers** – semantic search via `paraphrase-MiniLM-L3-v2`
- **PyMuPDF (fitz)** – PDF clause extraction
- **Railway** – Hosting and deployment

---

## API Endpoint

### `POST /hackrx/run`


