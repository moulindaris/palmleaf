"""Conservative Ancient Tamil to Modern Tamil suggestion support.

This module deliberately makes only documented lexical and orthographic
substitutions. It does not translate uncertain passages or claim that the
suggested wording is authoritative; historians must review every suggestion.
"""
from __future__ import annotations

import re
import unicodedata


# Classical / older forms that have an unambiguous, readable modern equivalent.
# Keep this compact and reviewable; additions should be backed by a source or an
# expert decision rather than guessed from OCR output.
MODERN_EQUIVALENTS = {
    "அஃது": "அது",
    "இஃது": "இது",
    "எமக்கு": "எங்களுக்கு",
    "எமது": "எங்களுடைய",
    "நுமக்கு": "உங்களுக்கு",
    "நுமது": "உங்களுடைய",
    "யான்": "நான்",
    "யாம்": "நாம்",
    "யாதும்": "எதுவும்",
    "யாவரும்": "அனைவரும்",
    "கேளிர்": "உறவினர்கள்",
    "ஈண்டு": "இங்கே",
    "அங்கு": "அங்கே",
    "எங்ஙனம்": "எப்படி",
    "யாங்கு": "எங்கே",
    "இவண்": "இவ்விடம்",
    "அவண்": "அவ்விடம்",
    "நின்": "உன்",
    "நின்னை": "உன்னை",
    "நினது": "உன்னுடைய",
    "உள": "உள்ள",
    "இல": "இல்லை",
    "அருளிச்": "அருளி",
}

# Longer expressions run before word substitutions so their intended meaning is
# not broken into individual tokens.
PHRASE_EQUIVALENTS = {
    "யாதும் ஊரே யாவரும் கேளிர்": "எல்லா ஊர்களும் எங்கள் ஊர்; அனைவரும் உறவினர்கள்",
    "தீதும் நன்றும் பிறர்தர வாரா": "தீமையும் நன்மையும் பிறரால் வருவதில்லை",
}

_WORD_PATTERN = re.compile(r"[\u0B80-\u0BFF]+")
_TAMIL_CHAR = r"\u0B80-\u0BFF"


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    """Match a known phrase across variable spacing, without partial words."""
    words = phrase.split()
    body = r"\s+".join(re.escape(word) for word in words)
    return re.compile(rf"(?<![{_TAMIL_CHAR}]){body}(?![{_TAMIL_CHAR}])")


def _reading_quality(text: str) -> tuple[bool, str]:
    """Identify OCR that is too fragmented for ethical normalization.

    This is a guardrail, not an OCR accuracy score. Ancient Tamil can contain
    rare vocabulary, so the test only blocks clearly fragmented output.
    """
    words = _WORD_PATTERN.findall(text)
    if len(words) < 3:
        return False, "Too little readable Tamil text was detected."
    fragmented = sum(len(word) <= 2 for word in words)
    if fragmented / len(words) >= 0.42:
        return False, "OCR text is highly fragmented; correct the OCR before modernizing it."
    return True, "OCR structure is sufficient for conservative rule suggestions."


def modernize_ancient_tamil(text: str) -> dict:
    """Return conservative phrase/word suggestions, even for fragmented OCR."""
    if not text or text.startswith("["):
        return {
            "modern_text": text,
            "changes": [],
            "confidence": 0.0,
            "status": "needs_ocr_review",
            "method": "No readable Tamil text was available for normalization.",
        }

    # Normalize canonically equivalent Tamil spellings before dictionary lookup.
    source_text = unicodedata.normalize("NFC", text)
    readable, quality_message = _reading_quality(source_text)

    changes: list[dict[str, str]] = []
    modern = source_text

    for source, target in PHRASE_EQUIVALENTS.items():
        pattern = _phrase_pattern(source)

        def replace_phrase(match: re.Match[str]) -> str:
            changes.append({"source": match.group(0), "target": target, "type": "phrase"})
            return target

        modern = pattern.sub(replace_phrase, modern)

    def replace_word(match: re.Match[str]) -> str:
        source = match.group(0)
        target = MODERN_EQUIVALENTS.get(source)
        if not target:
            return source
        changes.append({"source": source, "target": target, "type": "word"})
        return target

    modern = _WORD_PATTERN.sub(replace_word, modern)
    tamil_words = _WORD_PATTERN.findall(source_text)
    coverage = len(changes) / max(len(tamil_words), 1)
    # Conservative confidence: it measures only certainty of the explicit rule
    # changes, never historical authenticity or semantic correctness.
    confidence = 0.95 if changes else 0.0
    if changes and coverage > 0.45:
        confidence = 0.82
    if not readable:
        confidence = min(confidence, 0.35)

    return {
        "modern_text": modern,
        "changes": changes,
        "confidence": confidence,
        "coverage": round(min(coverage, 1.0), 3),
        "review_required": bool(changes),
        "status": ("needs_review" if not readable else
                   "suggestions_ready" if changes else "no_safe_changes"),
        "warning": "" if readable else quality_message,
        "method": "Rule-based fallback using a small reviewed lexicon and known phrases; historian review required.",
    }
