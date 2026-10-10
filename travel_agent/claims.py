"""Reproducible, explainable medical-insurance claim risk triage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DISCLAIMER = "Decision support only: this is not a fraud determination or medical advice. A qualified investigator must review every case."
FEATURE_NAMES = ["amount_10k", "prior_claims_30d", "provider_risk", "duplicate_invoice",
                 "missing_documents", "service_after_submission"]


def _features(claim: dict[str, Any]) -> list[float]:
    service_after = bool(
        claim.get("service_after_submission")
        or (claim.get("service_date") and claim.get("submission_date")
            and claim["service_date"] > claim["submission_date"])
    )
    return [
        float(claim.get("amount", 0) or 0) / 10000,
        float(claim.get("prior_claims_30d", 0) or 0),
        {"low": 0, "elevated": 1, "high": 2}.get(str(claim.get("provider_risk", "low")).lower(), 0),
        float(bool(claim.get("duplicate_invoice"))),
        float(bool(claim.get("missing_documents"))),
        float(service_after),
    ]


def _model():
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return None
    data = json.loads((Path(__file__).parent / "data" / "claims_training.json").read_text(encoding="utf-8"))
    x = [_features(row) for row in data]
    y = [row["fraud"] for row in data]
    model = LogisticRegression(random_state=7, solver="liblinear")
    model.fit(x, y)
    return model


MODEL = _model()


def _rule_reasons(claim: dict[str, Any]) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    if float(claim.get("amount", 0) or 0) >= 10000:
        score += 25
        reasons.append("High claim amount requires additional review.")
    if claim.get("prior_claims_30d", 0) >= 3:
        score += 25
        reasons.append("Several recent claims were reported within 30 days.")
    if claim.get("provider_risk") in {"high", "elevated"}:
        score += 30
        reasons.append("Provider risk flag is elevated.")
    if claim.get("duplicate_invoice"):
        score += 35
        reasons.append("Invoice resembles a previously submitted invoice.")
    if claim.get("missing_documents"):
        score += 10
        reasons.append("Supporting documents are incomplete.")
    if _features(claim)[-1]:
        score += 40
        reasons.append("Service date occurs after the submission date.")
    return score, reasons


def triage_claim(claim: dict[str, Any]) -> dict[str, Any]:
    rule_score, reasons = _rule_reasons(claim)
    features = _features(claim)
    probability = float(MODEL.predict_proba([features])[0][1]) if MODEL else rule_score / 100
    contributions = {}
    if MODEL:
        contributions = {
            name: round(float(value * coefficient), 4)
            for name, value, coefficient in zip(FEATURE_NAMES, features, MODEL.coef_[0])
            if value and value * coefficient > 0
        }
    for feature, contribution in sorted(contributions.items(), key=lambda item: item[1], reverse=True):
        label = feature.replace("_", " ")
        reason = f"{label.title()} increased model risk (contribution {contribution:.2f})."
        if reason not in reasons:
            reasons.append(reason)
    risk_score = min(100, max(round(probability * 100), min(rule_score, 100)))
    level = "high" if risk_score >= 60 else "medium" if risk_score >= 30 else "low"
    return {
        "risk_score": risk_score,
        "model_probability": round(probability, 4),
        "model": "sklearn_logistic_regression" if MODEL else "rules_fallback",
        "triage": level,
        "reasons": reasons or ["No configured rule or model feature was triggered."],
        "feature_contributions": contributions,
        "disclaimer": DISCLAIMER,
    }
