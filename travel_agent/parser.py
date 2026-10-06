"""Small, dependency-free parser for natural-language travel requests."""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Any

MONTHS = {
    name.lower(): index
    for index, name in enumerate(
        ["January", "February", "March", "April", "May", "June",
         "July", "August", "September", "October", "November", "December"], 1
    )
}
ACTIVITY_TERMS = (
    "hiking", "beach", "food", "museum", "culture", "history", "nightlife",
    "shopping", "nature", "wildlife", "adventure", "skiing", "temples",
    "photography", "relaxation",
)


def _parse_date(value: str, year_hint: int | None = None) -> date | None:
    value = value.strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d %B %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    match = re.fullmatch(r"(\d{1,2})\s+([A-Za-z]+)(?:\s+(\d{4}))?", value)
    if match and match.group(2).lower() in MONTHS:
        year = int(match.group(3) or year_hint or date.today().year)
        return date(year, MONTHS[match.group(2).lower()], int(match.group(1)))
    return None


def _extract_dates(text: str) -> tuple[date | None, date | None]:
    iso_dates = re.findall(r"\b\d{4}-\d{1,2}-\d{1,2}\b", text)
    if len(iso_dates) >= 2:
        return _parse_date(iso_dates[0]), _parse_date(iso_dates[1])
    range_match = re.search(
        r"\b(\d{1,2}\s+[A-Za-z]+(?:\s+\d{4})?)\s+(?:to|-|through)\s+"
        r"(\d{1,2}\s+[A-Za-z]+(?:\s+\d{4})?)\b", text, re.I
    )
    if range_match:
        start = _parse_date(range_match.group(1))
        end = _parse_date(range_match.group(2), start.year if start else None)
        return start, end
    duration = re.search(r"\b(\d+)\s*(?:day|days|night|nights)\b", text, re.I)
    single = re.search(r"\b(?:on|from|starting)\s+(\d{4}-\d{1,2}-\d{1,2})\b", text, re.I)
    if single:
        start = _parse_date(single.group(1))
        if start and duration:
            return start, start + timedelta(days=int(duration.group(1)))
    return None, None


def parse_travel_request(text: str) -> dict[str, Any]:
    """Extract intent, destination, dates, activities, and budget conservatively."""
    clean = " ".join((text or "").strip().split())
    lower = clean.lower()
    start, end = _extract_dates(clean)
    destination = None
    destination_match = re.search(
        r"\b(?:to|in|visit|visiting|around)\s+([A-Za-z][A-Za-z .'-]{1,40}?)(?=\s+(?:from|for|between|on|under|with|and|in)\b|[,.!?]|$)",
        clean, re.I,
    )
    if destination_match:
        destination = destination_match.group(1).strip(" .,-")
    budget = None
    budget_match = re.search(r"(?:under|budget(?: is)?|spend|maximum|max)\s*[$€£₹]?\s*([\d,]+)", lower)
    if budget_match:
        budget = float(budget_match.group(1).replace(",", ""))
    currency_match = re.search(r"([$€£₹]|USD|EUR|GBP|INR)", clean, re.I)
    currency = currency_match.group(1).upper() if currency_match else "USD"
    activities = [term for term in ACTIVITY_TERMS if re.search(rf"\b{re.escape(term)}\b", lower)]
    if any(word in lower for word in ("plan", "trip", "travel", "itinerary", "vacation")):
        intent = "plan_trip"
    elif any(word in lower for word in ("weather", "forecast")):
        intent = "check_weather"
    else:
        intent = "travel_help"
    return {
        "raw_text": clean,
        "intent": intent,
        "destination": destination,
        "start_date": start.isoformat() if start else None,
        "end_date": end.isoformat() if end else None,
        "activities": activities,
        "budget": budget,
        "currency": currency,
    }
