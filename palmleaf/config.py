import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "tamil-heritage-secret-key-2026")
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/tamil_manuscripts_db")
    UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
    RESTORED_FOLDER = BASE_DIR / "static" / "restored"
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "tif", "tiff"}

    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    RESTORED_FOLDER.mkdir(parents=True, exist_ok=True)
