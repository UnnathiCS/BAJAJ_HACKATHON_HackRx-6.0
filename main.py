from fastapi import FastAPI, Request, Header
from pydantic import BaseModel
import fitz  # PyMuPDF
import requests
import os
import tempfile
from typing import List
from sentence_transformers import SentenceTransformer, util

app = FastAPI()

# Load semantic model once
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

# ---------- Request Format ----------
class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

# ---------- Helper: Extract text from PDF ----------
def extract_clauses_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    clauses = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("blocks")
        for block in blocks:
            text = block[4].strip()
            if len(text.split()) > 6:
                clauses.append({
                    "page": page_num + 1,
                    "text": text.replace("\n", " ")
                })
    return clauses

# ---------- Helper: Answer a single question ----------
def answer_question(question, clauses, top_k=1):
    clause_texts = [c["text"] for c in clauses]
    clause_embeddings = model.encode(clause_texts, convert_to_tensor=True)
    query_embedding = model.encode(question, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, clause_embeddings, top_k=top_k)[0]
    top_clause = clauses[hits[0]["corpus_id"]]
    return top_clause["text"]

# ---------- Endpoint ----------
@app.post("/hackrx/run")
async def hackrx_runner(req: HackRxRequest, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        return {"error": "Unauthorized. Please provide a valid Bearer token."}

    try:
        # 1. Download PDF
        pdf_url = req.documents
        pdf_data = requests.get(pdf_url)
        tmp_pdf_path = tempfile.mktemp(suffix=".pdf")
        with open(tmp_pdf_path, "wb") as f:
            f.write(pdf_data.content)

        # 2. Extract Clauses
        clauses = extract_clauses_from_pdf(tmp_pdf_path)

        # 3. Answer Questions
        answers = []
        for q in req.questions:
            try:
                answer = answer_question(q, clauses)
                answers.append(answer)
            except:
                answers.append("Unable to find answer.")

        return {"answers": answers}

    except Exception as e:
        return {"error": f"Internal error: {str(e)}"}
