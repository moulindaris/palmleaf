import re

class TamilTextCleaner:
    CORRECTIONS = {
        r"ா\s+": "ா",
        r"\s+்": "்",
        r"\|": " ",
        r"[\x00-\x1F]": ""
    }

    @classmethod
    def clean(cls, raw_text: str) -> dict:
        cleaned = raw_text
        for pattern, replacement in cls.CORRECTIONS.items():
            cleaned = re.sub(pattern, replacement, cleaned)
        cleaned = " ".join(cleaned.split())
        return {
            "cleaned_text": cleaned,
            "correction_method": "Rule-based ligatures repair & Palm leaf crack removal",
            "keywords": cls.extract_keywords(cleaned)
        }

    @staticmethod
    def extract_keywords(text: str) -> list:
        tokens = text.split()
        return list({token for token in tokens if len(token) > 2})[:8]
