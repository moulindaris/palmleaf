"""Conservative OCR cleanup for Tamil text.

This stage repairs Unicode and layout damage without deleting punctuation or
making open-ended lexical guesses. The language model receives both the raw OCR
and this corrected reading so it can preserve uncertainty during modernization.
"""
from __future__ import annotations

import re
import unicodedata

_TAMIL = r"\u0B80-\u0BFF"
_TAMIL_WORD = re.compile(rf"[{_TAMIL}]+")
_TAMIL_MARK = re.compile(r"[\u0BBE-\u0BCD\u0B82\u0B83]")
_ZERO_WIDTH = re.compile(r"[\u200B-\u200F\uFEFF\u2060]")
_CONTROL = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
_BROKEN_MARK = re.compile(rf"([{_TAMIL}])\s+([{_TAMIL_MARK.pattern[1:-1]}])")
_REPEAT = re.compile(rf"([{_TAMIL}])\1{{4,}}")
# Arabic digits are commonly injected into palm-leaf OCR from image labels,
# page counters, or recognition noise. They are removed from the corrected
# reading (the Original OCR remains available for review). Tamil-script
# numerals are retained because they can be genuine manuscript notation.
_ARABIC_NUMBERS = re.compile(r"[0-9]+(?::[0-9]+)+|[0-9]+")

# Only merge split tokens when their joined form is known. This repairs common
# OCR splits while avoiding a global whitespace-removal pass that destroys word
# boundaries. Grow this list from reviewed PalmLeaf samples.
_KNOWN_JOINED_WORDS = {
    "அஃது", "இஃது", "எமக்கு", "எமது", "நுமக்கு", "நுமது", "யான்", "யாம்",
    "யாதும்", "யாவரும்", "கேளிர்", "ஈண்டு", "எங்ஙனம்", "யாங்கு", "இவண்",
    "அவண்", "நின்னை", "நினது", "அறம்", "தமிழ்", "மொழி", "மருந்து",
    "மூலிகை", "நூல்", "ஊரே", "தீதும்", "நன்றும்", "பிறர்தர", "வாரா",
    "கல்வி", "மக்கள்", "உலகம்", "உயிர்", "நல்லது", "வாழ்க", "வாழ்வார்",
}
_VALID_SHORT_WORDS = {"அ", "ஆ", "இ", "ஈ", "உ", "ஊ", "எ", "ஏ", "ஐ", "ஒ", "ஓ", "ஔ"}
_PUNCTUATION = r"[,;:.!?()\[\]{}<>|।॥—–…'\"“”‘’]"


def _join_known_split_words(line: str, changes: list[dict[str, str]]) -> str:
    """Join adjacent Tamil fragments only if their concatenation is reviewed."""
    tokens = re.findall(rf"[{_TAMIL}]+|\s+|{_PUNCTUATION}|[^\s{_TAMIL}]", line)
    i = 0
    while i + 2 < len(tokens):
        left, separator, right = tokens[i:i + 3]
        if (_TAMIL_WORD.fullmatch(left) and separator.isspace()
                and _TAMIL_WORD.fullmatch(right) and left + right in _KNOWN_JOINED_WORDS):
            joined = left + right
            changes.append({"source": f"{left} {right}", "target": joined, "type": "joined_word"})
            tokens[i:i + 3] = [joined]
            i = max(i - 2, 0)
        else:
            i += 1
    return "".join(tokens)


def correct_tamil_ocr(raw: str) -> dict:
    """Return a recoverable corrected reading, repair log, and uncertain tokens."""
    if raw is None:
        raw = ""
    if not isinstance(raw, str):
        raw = str(raw)

    text = unicodedata.normalize("NFC", raw)
    text = _ZERO_WIDTH.sub("", text)
    text = _CONTROL.sub("", text)
    changes: list[dict[str, str]] = []
    removed_numbers = _ARABIC_NUMBERS.findall(text)
    if removed_numbers:
        text = _ARABIC_NUMBERS.sub("", text)
        changes.append({
            "source": ", ".join(removed_numbers[:10]),
            "target": "[removed Arabic numerals; review Original OCR]",
            "type": "numeric_artifact",
        })

    # OCR sometimes separates a dependent vowel mark from its base character.
    text, mark_count = _BROKEN_MARK.subn(r"\1\2", text)
    if mark_count:
        changes.append({"source": "split Tamil letter mark", "target": "joined Unicode grapheme", "type": "unicode"})

    # Keep line breaks and punctuation; normalize only spaces around punctuation.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([,;:.!?।॥…])", r"\1", text)
    text = re.sub(rf"([,;:.!?।॥])(?=[{_TAMIL}])", r"\1 ", text)

    # A bar between Tamil text is a common OCR rendering of a verse separator.
    text = re.sub(rf"(?<=[{_TAMIL}])\s*\|\s*(?=[{_TAMIL}])", " । ", text)

    corrected_lines = []
    for line in text.splitlines():
        line = _join_known_split_words(line, changes)
        # Do not collapse the actual content or punctuation; only trim line edges.
        corrected_lines.append(line.strip())
    text = "\n".join(line for line in corrected_lines if line)

    # Excessive repeated Tamil glyphs are often OCR streaks. Retain a doubled
    # trace for review rather than deleting the token.
    text, repeats = _REPEAT.subn(r"\1\1", text)
    if repeats:
        changes.append({"source": "repeated OCR glyphs", "target": "collapsed repeated run", "type": "repeat"})

    uncertain = []
    seen = set()
    for token in _TAMIL_WORD.findall(text):
        if (len(token) <= 2 and token not in _VALID_SHORT_WORDS) or "�" in token:
            if token not in seen:
                seen.add(token)
                uncertain.append(token)
    if "�" in text and "�" not in seen:
        uncertain.append("�")
    if re.search(rf"[{_TAMIL}][A-Za-z]|[A-Za-z][{_TAMIL}]", text):
        uncertain.append("mixed Tamil/Latin OCR")

    source_tamil = len(_TAMIL_WORD.findall(raw))
    corrected_tamil = len(_TAMIL_WORD.findall(text))
    confidence = 0.82 if corrected_tamil and not uncertain else (0.55 if corrected_tamil else 0.0)
    if changes:
        confidence = min(confidence, 0.80)
    warnings = []
    if uncertain:
        warnings.append("Some short or mixed-script tokens may be OCR errors. Review the highlighted items.")
    if removed_numbers:
        warnings.append(
            "Removed Arabic digits as probable OCR artifacts; check Original OCR "
            "if they were meaningful verse or page numbers. Tamil-script numerals are preserved."
        )
    warning = " ".join(warnings)
    if not text:
        warning = "No Tamil characters were recovered. Try OCR again or enter a correction manually."

    return {
        "corrected_text": text,
        "changes": changes,
        "uncertain_words": uncertain,
        "confidence": confidence,
        "confidence_kind": "estimated OCR quality, not a calibrated probability",
        "warning": warning,
        "status": "needs_review" if uncertain or not text else "corrected",
        "source_tamil_tokens": source_tamil,
    }


def clean_tamil_text(raw: str) -> str:
    """Backward-compatible string-only wrapper used by the legacy OCR model."""
    result = correct_tamil_ocr(raw)
    return result["corrected_text"] or "[No readable Tamil text found]"


def diff_text(raw: str, cleaned: str) -> dict:
    """Return character-level stats for before/after comparison."""
    return {
        "raw_chars": len(raw),
        "clean_chars": len(cleaned),
        "removed_chars": max(0, len(raw) - len(cleaned)),
        "raw_lines": len(raw.splitlines()),
        "clean_lines": len(cleaned.splitlines()),
    }
