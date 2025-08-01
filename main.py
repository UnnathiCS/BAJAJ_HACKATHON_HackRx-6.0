from fastapi import FastAPI, Header
from pydantic import BaseModel
import fitz  # PyMuPDF
import requests
import tempfile
from typing import List
from sentence_transformers import SentenceTransformer, util
from functools import lru_cache
import torch
torch.set_num_threads(1)

app = FastAPI()

@lru_cache(maxsize=1)
def get_model():
    return SentenceTransformer("paraphrase-MiniLM-L3-v2")

class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

def extract_clauses_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    clauses = []
    for page_num in range(min(len(doc), 25)):  # limit to 25 pages
        blocks = doc[page_num].get_text("blocks")
        for block in blocks:
            text = block[4].strip()
            if len(text.split()) > 8:  # avoid tiny fragments
                clauses.append({
                    "page": page_num + 1,
                    "text": text.replace("\n", " ")
                })
    return clauses

def answer_question(question, clauses, top_k=3):
    model = get_model()
    clause_texts = [c["text"] for c in clauses]
    clause_embeddings = model.encode(clause_texts, convert_to_tensor=True)
    q_embedding = model.encode(question, convert_to_tensor=True)
    hits = util.semantic_search(q_embedding, clause_embeddings, top_k=top_k)[0]

    if hits and hits[0]["score"] > 0.3:
        best_clause = clauses[hits[0]["corpus_id"]]["text"]

        # 🎯 Extract the most relevant sentence
        question_keywords = set(question.lower().split())
        best_sent = max(
            best_clause.split("."), 
            key=lambda s: len(set(s.lower().split()) & question_keywords)
        )

        return best_sent.strip() + "."  # always end with period
    else:
        return "Unable to find answer."



@app.get("/")
def root():
    return {"message": "HackRx API is live ✅"}

@app.get("/hackrx/run")
def fallback_run():
    return {"message": "Please send a POST request to this endpoint with insurance PDF and questions."}

@app.post("/hackrx/run")
async def hackrx_runner(req: HackRxRequest, authorization: str = Header(None)):
    try:
        pdf_url = req.documents
        response = requests.get(pdf_url)
        tmp_pdf_path = tempfile.mktemp(suffix=".pdf")
        with open(tmp_pdf_path, "wb") as f:
            f.write(response.content)

        clauses = extract_clauses_from_pdf(tmp_pdf_path)

        answers = []
        for q in req.questions:
            try:
                a = answer_question(q, clauses)
                answers.append(a)
            except Exception:
                answers.append("Unable to find answer.")

        return {"answers": answers}

    except Exception as e:
        return {"error": f"Internal error: {str(e)}"}
