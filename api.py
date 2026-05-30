"""
LendMN Credit Scoring — FastAPI Inference API
Run: uvicorn src.api:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Literal
import joblib
import numpy as np
from pathlib import Path

ROOT   = Path(__file__).parent
MODELS = ROOT / "models"

app = FastAPI(
    title="LendMN Credit Scoring API",
    description="ML-powered credit scoring for Mongolian lending market",
    version="1.0.0",
)

# Load artifacts at startup
model        = joblib.load(MODELS / "best_model.joblib")
le           = joblib.load(MODELS / "label_encoder.joblib")
feature_cols = joblib.load(MODELS / "feature_cols.joblib")


class ApplicantIn(BaseModel):
    age:                  int   = Field(..., ge=18, le=70,         description="Нас")
    employment_type:      Literal["government","private","self_employed","freelance"] = "private"
    monthly_income:       int   = Field(..., ge=300_000,           description="Сарын орлого (₮)")
    employment_months:    int   = Field(..., ge=0, le=360,         description="Ажилласан хугацаа (сар)")
    num_existing_loans:   int   = Field(..., ge=0, le=10,          description="Одоогийн зээлийн тоо")
    credit_history_months:int   = Field(..., ge=0,                 description="Зээлийн түүх (сар)")
    previous_defaults:    int   = Field(..., ge=0,                 description="Өмнөх хугацаа хэтрэлт")
    num_dependents:       int   = Field(..., ge=0,                 description="Тэжээлд авах хүний тоо")
    loan_amount:          int   = Field(..., ge=100_000,           description="Зээлийн дүн (₮)")


class ScoreOut(BaseModel):
    credit_score:      int
    score_band:        str
    default_probability: float
    recommendation:    str
    risk_level:        str
    debt_to_income:    float
    loan_to_income:    float


def prob_to_score(prob: float) -> int:
    return int(np.clip(850 - (prob ** 0.5) * 550, 300, 850))


def score_band(score: int) -> str:
    if score >= 750: return "Маш сайн"
    if score >= 700: return "Сайн"
    if score >= 650: return "Дундаж"
    if score >= 580: return "Дор"
    return "Маш дор"


def risk_level(score: int) -> str:
    if score >= 700: return "LOW"
    if score >= 580: return "MEDIUM"
    return "HIGH"


def recommendation(score: int) -> str:
    if score >= 700: return "Зээл олгохыг зөвлөж байна."
    if score >= 580: return "Нэмэлт баримт бичиг шаардлагатай."
    return "Зээл олгохыг зөвлөхгүй."


@app.post("/score", response_model=ScoreOut, summary="Credit score тооцоолох")
def score_applicant(data: ApplicantIn):
    emp_enc = le.transform([data.employment_type])[0]
    dti = (data.loan_amount * 0.05) / data.monthly_income
    lti = data.loan_amount / (data.monthly_income * 12)

    X = np.array([[
        data.age, emp_enc, data.monthly_income, data.employment_months,
        data.num_existing_loans, data.credit_history_months,
        data.previous_defaults, data.num_dependents,
        data.loan_amount, dti, lti,
    ]])

    prob  = float(model.predict_proba(X)[0][1])
    score = prob_to_score(prob)

    return ScoreOut(
        credit_score        = score,
        score_band          = score_band(score),
        default_probability = round(prob, 4),
        recommendation      = recommendation(score),
        risk_level          = risk_level(score),
        debt_to_income      = round(dti, 4),
        loan_to_income      = round(lti, 4),
    )


@app.get("/health")
def health():
    return {"status": "ok", "model": "Logistic Regression", "auc": 0.9939}
