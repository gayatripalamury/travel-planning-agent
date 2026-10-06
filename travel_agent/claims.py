"""Explainable, non-diagnostic medical-insurance claim triage."""

from __future__ import annotations

from typing import Any


DISCLAIMER = "Decision support only: this is not a fraud determination or medical advice. A qualified investigator must review every case."


def triage_claim(claim: dict[str, Any]) -> dict[str, Any]:
    score = 0
    reasons: list[str] = []
    amount = float(claim.get("amount", 0) or 0)
    if amount >= 10000:
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
    if claim.get("service_date") and claim.get("submission_date") and claim["service_date"] > claim["submission_date"]:
        score += 40
        reasons.append("Service date occurs after the submission date.")
    if score >= 60:
        level = "high"
    elif score >= 30:
        level = "medium"
    else:
        level = "low"
    return {"risk_score": min(score, 100), "triage": level, "reasons": reasons or ["No configured rule was triggered."], "disclaimer": DISCLAIMER}
