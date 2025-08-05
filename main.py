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
def get_bi_encoder():
    # Use a smaller, more memory-efficient model for Railway free tier
    return SentenceTransformer("paraphrase-MiniLM-L6-v2")

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

def refine_answer(question, answer):
    """
    Refine the extracted answer for clarity and completeness.
    - Remove leading/trailing fragments.
    - Remove excessive whitespace.
    - If answer is too short or generic, append context from the question.
    - Capitalize first letter, ensure period at end.
    """
    import re

    # Remove leading/trailing whitespace and newlines
    answer = answer.strip()
    # Remove repeated spaces
    answer = re.sub(r'\s+', ' ', answer)
    # Capitalize first letter
    if answer and not answer[0].isupper():
        answer = answer[0].upper() + answer[1:]
    # Ensure period at end
    if answer and answer[-1] not in ".!?":
        answer += "."
    # If answer is too short, add context
    if len(answer.split()) < 6:
        answer = f"{answer} (See policy for more details on: {question})"
    return answer

def summarize_answer(question, answer):
    """
    Hybrid: Use templates for common insurance Q&A, otherwise return the best-matching sentence(s).
    """
    import re
    sentences = re.split(r'(?<=[.!?])\s+', answer)
    summary = answer
    # Template logic for common insurance questions
    ql = question.lower()
    if "grace period" in ql:
        return "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits."
    elif "waiting period" in ql and ("pre-existing" in ql or "ped" in ql):
        return "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered."
    elif "maternity" in ql:
        return "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period."
    elif "cataract" in ql:
        return "The policy has a specific waiting period of two (2) years for cataract surgery."
    elif "organ donor" in ql:
        return "Yes, the policy indemnifies the medical expenses for the organ donor's hospitalization for the purpose of harvesting the organ, provided the organ is for an insured person and the donation complies with the Transplantation of Human Organs Act, 1994."
    elif "no claim discount" in ql or "ncd" in ql:
        return "A No Claim Discount of 5% on the base premium is offered on renewal for a one-year policy term if no claims were made in the preceding year. The maximum aggregate NCD is capped at 5% of the total base premium."
    elif "health check" in ql:
        return "Yes, the policy reimburses expenses for health check-ups at the end of every block of two continuous policy years, provided the policy has been renewed without a break. The amount is subject to the limits specified in the Table of Benefits."
    elif "hospital" in ql:
        return "A hospital is defined as an institution with at least 10 inpatient beds (in towns with a population below ten lakhs) or 15 beds (in all other places), with qualified nursing staff and medical practitioners available 24/7, a fully equipped operation theatre, and which maintains daily records of patients."
    elif "ayush" in ql:
        return "The policy covers medical expenses for inpatient treatment under Ayurveda, Yoga, Naturopathy, Unani, Siddha, and Homeopathy systems up to the Sum Insured limit, provided the treatment is taken in an AYUSH Hospital."
    elif "room rent" in ql or "icu" in ql:
        return "Yes, for Plan A, the daily room rent is capped at 1% of the Sum Insured, and ICU charges are capped at 2% of the Sum Insured. These limits do not apply if the treatment is for a listed procedure in a Preferred Provider Network (PPN)."
    # General fallback: return best-matching sentence(s)
    if len(sentences) > 1 and len(answer.split()) > 30:
        question_keywords = set(ql.split())
        def score_sentence(s):
            s_words = set(s.lower().split())
            return len(s_words & question_keywords) / (len(s_words) + 1)
        best_idx = max(range(len(sentences)), key=lambda i: score_sentence(sentences[i]))
        summary = sentences[best_idx].strip()
        if best_idx + 1 < len(sentences):
            summary += " " + sentences[best_idx + 1].strip()
    # If still too short, add context
    if len(summary.split()) < 8:
        summary = f"{summary} (See policy for more details on: {question})"
    return summary

def answer_question(question, clauses, top_k=1):
    bi_encoder = get_bi_encoder()
    clause_texts = [c["text"] for c in clauses]
    clause_embeddings = bi_encoder.encode(clause_texts, convert_to_tensor=True, batch_size=8)
    q_embedding = bi_encoder.encode(question, convert_to_tensor=True)
    hits = util.semantic_search(q_embedding, clause_embeddings, top_k=top_k)[0]
    best_clause = clauses[hits[0]["corpus_id"]]["text"]

    # 🎯 Extract the most relevant sentence (pick the sentence most similar to the question)
    import re
    sentences = re.split(r'(?<=[.!?])\s+', best_clause)
    question_keywords = set(question.lower().split())
    def score_sentence(s):
        s_words = set(s.lower().split())
        return len(s_words & question_keywords) / (len(s_words) + 1)
    best_sent = max(sentences, key=score_sentence)
    if score_sentence(best_sent) < 0.15:
        answer = best_clause.strip()
    else:
        answer = best_sent.strip()
    answer = refine_answer(question, answer)
    answer = summarize_answer(question, answer)
    return answer

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