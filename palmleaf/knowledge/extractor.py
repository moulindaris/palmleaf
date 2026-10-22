"""
knowledge/extractor.py
Phase 8 — Keyword-based knowledge extraction and category tagging.
"""

# Category → keywords mapping (Tamil + transliterated)
KNOWLEDGE_MAP = {
    "Medicine": [
        "மருந்து", "வைத்தியம்", "நோய்", "மூலிகை", "சிகிச்சை",
        "உடல்", "ஆரோக்கியம்", "கஷாயம்", "தைலம்", "பத்தியம்",
        "medicine", "herb", "cure", "treatment", "disease"
    ],
    "Astrology": [
        "ஜோதிடம்", "நட்சத்திரம்", "கிரகம்", "ராசி", "பஞ்சாங்கம்",
        "சோதிடம்", "லக்னம்", "திதி", "யோகம்", "கரணம்",
        "astrology", "star", "planet", "horoscope", "zodiac"
    ],
    "Literature": [
        "கவிதை", "பாடல்", "இலக்கியம்", "சங்கம்", "தமிழ்",
        "செய்யுள்", "நூல்", "காப்பியம்", "அகம்", "புறம்",
        "poem", "verse", "literature", "song", "epic"
    ],
    "Philosophy": [
        "தத்துவம்", "ஆன்மா", "வேதம்", "சித்தாந்தம்", "மோட்சம்",
        "கடவுள்", "பக்தி", "யோகம்", "ஞானம்", "சைவம்",
        "philosophy", "soul", "god", "devotion", "wisdom"
    ],
    "Mathematics": [
        "கணிதம்", "எண்", "கணக்கு", "அளவு", "வடிவியல்",
        "math", "number", "calculation", "geometry", "measure"
    ],
    "History": [
        "வரலாறு", "அரசன்", "சோழன்", "பாண்டியன்", "சேரன்",
        "போர்", "கோட்டை", "நாடு", "ஆட்சி", "வெற்றி",
        "history", "king", "war", "dynasty", "kingdom"
    ],
}

DYNASTY_KEYWORDS = {
    "Chola":   ["சோழன்", "சோழர்", "chola", "kulottunga", "rajaraja"],
    "Pandya":  ["பாண்டியன்", "பாண்டியர்", "pandya", "madurai"],
    "Chera":   ["சேரன்", "சேரர்", "chera", "kerala"],
    "Pallava": ["பல்லவன்", "பல்லவர்", "pallava", "kanchipuram"],
    "Nayak":   ["நாயக்கர்", "nayak", "vijayanagara"],
}

PERIOD_MAP = {
    "Chola":   "9th–13th Century CE",
    "Pandya":  "6th–14th Century CE",
    "Chera":   "1st–5th Century CE",
    "Pallava": "4th–9th Century CE",
    "Nayak":   "16th–18th Century CE",
    "Unknown": "Unknown Period",
}


def extract_knowledge(text: str) -> dict:
    """
    Returns:
      - tags: list of (tag_word, category)
      - category: dominant category
      - dynasty: detected dynasty
      - period: estimated period
    """
    if not text or text.startswith("["):
        return {"tags": [], "category": "Unknown", "dynasty": "Unknown", "period": "Unknown"}

    lower = text.lower()
    tag_hits = []
    category_scores = {cat: 0 for cat in KNOWLEDGE_MAP}

    for category, keywords in KNOWLEDGE_MAP.items():
        for kw in keywords:
            if kw.lower() in lower:
                tag_hits.append((kw, category))
                category_scores[category] += 1

    # Dominant category
    dominant = max(category_scores, key=category_scores.get)
    if category_scores[dominant] == 0:
        dominant = "Unknown"

    # Dynasty detection
    detected_dynasty = "Unknown"
    for dynasty, keywords in DYNASTY_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in lower:
                detected_dynasty = dynasty
                break
        if detected_dynasty != "Unknown":
            break

    period = PERIOD_MAP.get(detected_dynasty, "Unknown Period")

    return {
        "tags":     tag_hits,
        "category": dominant,
        "dynasty":  detected_dynasty,
        "period":   period,
    }
