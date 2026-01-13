from datetime import datetime

from dateparser.search import search_dates

text = "Patient felt nausea 2 days after the second infusion."
ref_date = datetime(2024, 1, 1)

settings = {
    "RELATIVE_BASE": ref_date,
    "RETURN_AS_TIMEZONE_AWARE": True,
    "PREFER_DATES_FROM": "past",
}

print(f"Text: {text}")
results = search_dates(text, languages=["en"], settings=settings)
print(f"Results: {results}")

text2 = "3 days after admission"
print(f"Text: {text2}")
results2 = search_dates(text2, languages=["en"], settings=settings)
print(f"Results: {results2}")
