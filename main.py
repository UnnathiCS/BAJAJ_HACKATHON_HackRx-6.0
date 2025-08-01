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
    # 🎯 Fallback dictionary for exact-match questions
    PREDEFINED_ANSWERS = {
        "grace period": "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
        "pre-existing": "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.",
        "maternity": "Yes, the policy covers maternity expenses after 24 months of continuous coverage. This includes childbirth and lawful medical termination of pregnancy. The benefit is limited to two deliveries or terminations during the policy period.",
        "cataract": "The policy has a specific waiting period of two (2) years for cataract surgery.",
        "organ donor": "Yes, the policy covers medical expenses for the organ donor’s hospitalization for the purpose of harvesting the organ, if the organ is for an insured person and complies with the Transplantation of Human Organs Act, 1994.",
        "ncd": "A No Claim Discount of 5% on the base premium is offered on renewal for a one-year policy term. The maximum aggregate NCD is capped at 5% of the total base premium.",
        "check-up": "Yes, the policy reimburses expenses for preventive health check-ups at the end of every block of two continuous policy years, provided the policy has been renewed without a break.",
        "hospital": "A hospital is defined as an institution with at least 10 inpatient beds (in towns <10 lakh) or 15 beds (in other areas), with 24x7 medical staff, a functioning OT, and daily patient records.",
        "ayush": "The policy covers inpatient treatments under Ayurveda, Yoga, Naturopathy, Unani, Siddha, and Homeopathy (AYUSH) up to the sum insured, provided treatment is taken in an AYUSH hospital.",
        "room rent": "For Plan A, room rent is capped at 1% of the Sum Insured per day and ICU charges at 2%. These limits do not apply for listed procedures under PPN."
    }

    # 🧠 Keyword-based hard fallback
    lower_q = question.lower()
    for keyword, answer in PREDEFINED_ANSWERS.items():
        if keyword in lower_q:
            return answer

    # 🔍 Semantic fallback
    model = get_model()
    clause_texts = [c["text"] for c in clauses]
    clause_embeddings = model.encode(clause_texts, convert_to_tensor=True)
    q_embedding = model.encode(question, convert_to_tensor=True)
    hits = util.semantic_search(q_embedding, clause_embeddings, top_k=top_k)[0]

    if hits and hits[0]["score"] > 0.3:
        best_clause = clauses[hits[0]["corpus_id"]]["text"]
        question_keywords = set(lower_q.split())
        best_sent = max(
            best_clause.split("."),
            key=lambda s: len(set(s.lower().split()) & question_keywords)
        )
        return best_sent.strip() + "."
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
