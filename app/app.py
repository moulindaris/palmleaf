# -*- coding: utf-8 -*-
"""
app/app.py
Full Tamil Palm-Leaf Manuscript Digitizer & Intelligent Analysis System.
Features User Authentication, Interactive Image Slider, CSV Export, Comments & Admin Control Dashboard.
"""
import os
import sys
import csv
from threading import Lock
from io import StringIO
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import torch
from flask import Flask, abort, render_template, request, jsonify, send_from_directory, session, redirect, url_for, Response
from werkzeug.utils import secure_filename
from config import Config

from preprocessing.preprocess import preprocess_image
from models.restoration_model import get_model, load_weights as load_unet_weights
from ocr.ocr import extract_text
from utils.helpers import load_image, save_image, to_base64, restore_image
from utils.text_cleaner import correct_tamil_ocr, diff_text
from utils.tamil_modernizer import modernize_tamil
from knowledge.extractor import extract_knowledge
from classification.classifier import classify_manuscript
from training.inference import (
    restore_image_ml, classify_image_ml, get_model_status
)
from database.db import (
    init_db, save_manuscript, update_manuscript, get_manuscript,
    search_manuscripts, get_all_manuscripts, get_tdl_catalog_records,
    get_cict_catalog_records, save_artifact,
    get_artifacts, save_tags, get_tags, save_achievement,
    get_analytics, get_locations, get_achievements, delete_manuscript,
    create_user, verify_user, get_all_users, add_comment, get_comments,
    log_action, get_system_logs
)
from utils.exporter import build_html_report, build_text_report

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff", "pdf"}
PDF_RENDER_LOCK = Lock()
MODEL_PATH = os.path.join(ROOT, "models", "restorer.pth")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB
app.secret_key = Config.SECRET_KEY

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------------------------------------------------------------
# Init DB & Models
# ---------------------------------------------------------------------------
init_db()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = get_model(DEVICE)
if os.path.exists(MODEL_PATH):
    load_unet_weights(model, MODEL_PATH, DEVICE)
    print(f"[INFO] Model loaded from {MODEL_PATH} on {DEVICE}")
else:
    print("[WARN] No trained model found. Run: python training/train_tamil.py")
model.eval()

# Pre-load all ML models at startup
from training.inference import _load_restorer, _load_classifier, _load_corrector
_load_restorer()
_load_classifier()
_load_corrector()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.context_processor
def inject_user():
    return dict(current_user=session.get("user"))


# Redirect legacy API-docs route to index
@app.route("/api-docs")
def api_docs_redirect():
    return redirect(url_for("index"))


# ---------------------------------------------------------------------------
# User Authentication & Profile Routes
# ---------------------------------------------------------------------------

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if request.form.get("mode") == "guest":
            session["user"] = {
                "id": None,
                "username": "Guest",
                "email": "",
                "role": "Visitor",
                "is_guest": True,
            }
            return redirect(url_for("index"))

        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        user = verify_user(username, password)
        if user:
            session["user"] = {
                "id": user["id"],
                "username": user["username"],
                "email": user["email"],
                "role": user["role"]
            }
            log_action("User Login", username, "Logged into system")
            return redirect(url_for("index"))
        else:
            return render_template("login.html", error="Invalid username or password")
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        role = request.form.get("role", "Student / Researcher")

        if not username or not email or not password:
            return render_template("register.html", error="All fields are required")

        success = create_user(username, email, password, role)
        if success:
            user = verify_user(username, password)
            session["user"] = {
                "id": user["id"],
                "username": user["username"],
                "email": user["email"],
                "role": user["role"]
            }
            return redirect(url_for("index"))
        else:
            return render_template("register.html", error="Username or Email already exists")
    return render_template("register.html")


@app.route("/logout")
def logout():
    user = session.get("user")
    if user:
        log_action("User Logout", user["username"], "Logged out")
    session.pop("user", None)
    return redirect(url_for("index"))


@app.route("/profile")
def profile():
    if "user" not in session or session["user"].get("is_guest"):
        return redirect(url_for("login"))
    user = session["user"]
    logs = get_system_logs(limit=20)
    user_logs = [l for l in logs if l["user_name"] == user["username"]]
    return render_template("profile.html", user=user, logs=user_logs)


@app.route("/admin")
def admin():
    if "user" not in session or session["user"]["role"] != "Administrator":
        return "Unauthorized Access. Administrator role required.", 403

    users = get_all_users()
    logs = get_system_logs(limit=100)
    manuscripts = get_all_manuscripts()
    return render_template("admin.html", users=users, logs=logs, manuscripts=manuscripts)


# ---------------------------------------------------------------------------
# Page Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/repository")
def repository():
    manuscripts = get_all_manuscripts()
    catalog_records = get_tdl_catalog_records()
    cict_catalog_records = get_cict_catalog_records()
    catalog_facets = sorted({
        str(value).strip()
        for record in cict_catalog_records
        for value in (record.get("collection"),)
        if value
    } | {
        str(value).strip()
        for record in catalog_records
        for value in (record.get("subject"), record.get("record_type"))
        if value
    }, key=str.casefold)
    learning_routes = [
        {
            "title": "Begin with short works",
            "level": "START HERE",
            "description": "Read concise verses and ethical ideas before exploring longer texts.",
            "works": [
                ("ஆத்திசூடி", "Aathichudi", "Short moral sayings arranged for easy reading."),
                ("திருக்குறள்", "Thirukkural", "Couplets on virtue, society, and love."),
                ("நாலடியார்", "Naladiyar", "Four-line poems with ethical reflections."),
            ],
        },
        {
            "title": "Explore Sangam poetry",
            "level": "NEXT PATH",
            "description": "Meet early Tamil poems about public life, kingship, war, and generosity.",
            "works": [("புறநானூறு", "Purananuru", "A collection of poems on public life and heroism.")],
        },
        {
            "title": "Read the Tamil epics",
            "level": "NEXT PATH",
            "description": "Follow longer narrative poems and their connected stories.",
            "works": [
                ("சிலப்பதிகாரம்", "Silappathikaram", "An influential Tamil epic centered on Kannagi and Kovalan."),
                ("மணிமேகலை", "Manimekalai", "An epic that continues a story world linked to Silappathikaram."),
            ],
        },
        {
            "title": "Continue into devotional writing",
            "level": "ANOTHER ROUTE",
            "description": "Explore spiritual and philosophical writing in Tamil.",
            "works": [("திருமந்திரம்", "Thirumanthiram", "A Saiva work combining devotion and philosophy.")],
        },
    ]
    for route in learning_routes:
        linked_works = []
        for tamil_title, english_title, description in route["works"]:
            record = next((item for item in catalog_records if tamil_title in item.get("title", "")), None)
            linked_works.append({
                "tamil_title": tamil_title,
                "english_title": english_title,
                "description": description,
                "source_id": record.get("source_id") if record else None,
                "source_url": record.get("source_url") if record else None,
            })
        route["works"] = linked_works
    return render_template(
        "repository.html", manuscripts=manuscripts,
        catalog_records=catalog_records, cict_catalog_records=cict_catalog_records,
        learning_routes=learning_routes, catalog_facets=catalog_facets,
    )


@app.route("/manuscript/<int:mid>")
def manuscript_detail(mid):
    m = get_manuscript(mid)
    if not m:
        return render_template("404.html"), 404
    if m.get("demo_record"):
        return render_template("catalog_sample_detail.html", m=m)
    artifacts = get_artifacts(mid)
    tags = get_tags(mid)
    comments = get_comments(mid)
    raw_text = m.get("raw_text") or m.get("clean_text") or ""
    correction = correct_tamil_ocr(raw_text)
    return render_template(
        "detail.html", m=m, artifacts=artifacts, tags=tags, comments=comments,
        correction=correction, original_ocr=raw_text,
    )


@app.route("/literature/<source_id>")
def literature_detail(source_id):
    record = next((item for item in get_tdl_catalog_records()
                   if item.get("source_id") == source_id), None)
    if not record:
        record = next((item for item in get_cict_catalog_records()
                       if item.get("source_id") == source_id), None)
    if not record or record.get("record_type") != "literary_work":
        abort(404)
    work_title = record.get("title", "").lower()
    video_id = "VMmNb3K63BI" if any(
        title in work_title for title in ("திருக்குறள்", "tirukkural", "tirukkuṟaḷ")
    ) else None
    return render_template("literature_detail.html", record=record, video_id=video_id)


def _normalise(value):
    """Create a safe, case-insensitive value for catalogue filtering."""
    return str(value or "").casefold()


def _manuscript_matches(manuscript, keyword, category, dynasty, period, script, location):
    """Apply the catalogue filters to a manuscript record and its knowledge tags."""
    if category != "All" and _normalise(manuscript.get("category")) != _normalise(category):
        return False
    if dynasty != "All" and _normalise(dynasty) not in _normalise(manuscript.get("dynasty")):
        return False
    if period != "All" and _normalise(period) not in _normalise(manuscript.get("period")):
        return False
    if script != "All" and _normalise(script) not in _normalise(manuscript.get("style")):
        return False
    if location and _normalise(location) not in _normalise(manuscript.get("location")):
        return False

    if not keyword:
        return True

    searchable = " ".join(str(manuscript.get(field, "")) for field in (
        "title", "raw_text", "clean_text", "category", "dynasty", "period", "style", "location"
    ))
    searchable += " " + " ".join(
        f"{tag.get('tag', '')} {tag.get('category', '')}" for tag in get_tags(manuscript["id"])
    )
    return _normalise(keyword) in _normalise(searchable)


@app.route("/search")
def search():
    keyword = request.args.get("q", "").strip()
    category = request.args.get("category", "All")
    dynasty = request.args.get("dynasty", "All")
    period = request.args.get("period", "All")
    script = request.args.get("script", "All")
    location = request.args.get("location", "").strip()

    # Filtering in this layer keeps the UI independent of the eventual database
    # implementation (SQLite, MongoDB, or an external catalogue API).
    results = [m for m in get_all_manuscripts() if _manuscript_matches(
        m, keyword, category, dynasty, period, script, location
    )]
    return render_template(
        "search.html", results=results, keyword=keyword, category=category,
        dynasty=dynasty, period=period, script=script, location=location
    )


@app.route("/map")
def map_view():
    # The basemap is modern geography. Keep period-aware names in the historic
    # layer and use approximate city coordinates only when exact coordinates
    # were not recorded with the manuscript.
    known_coordinates = {
        "thanjavur": (10.7870, 79.1378),
        "madurai": (9.9252, 78.1198),
        "kanchipuram": (12.8342, 79.7036),
        "chennai": (13.0827, 80.2707),
        "chidambaram": (11.3992, 79.6936),
        "tirunelveli": (8.7139, 77.7567),
        "coimbatore": (11.0168, 76.9558),
        "pudukkottai": (10.3797, 78.8202),
        "tiruchirappalli": (10.7905, 78.7047),
        "ramanathapuram": (9.3639, 78.8395),
    }
    locations = []
    for manuscript in get_all_manuscripts():
        site = manuscript.get("discovery_location") or manuscript.get("location") or ""
        if isinstance(site, dict):
            site_name = site.get("location_name") or site.get("name") or "Unknown site"
            latitude = site.get("latitude")
            longitude = site.get("longitude")
        else:
            site_name = str(site)
            latitude = manuscript.get("latitude")
            longitude = manuscript.get("longitude")

        if latitude is None or longitude is None:
            place_key = next((key for key in known_coordinates if key in site_name.casefold()), None)
            if place_key:
                latitude, longitude = known_coordinates[place_key]
        if latitude is None or longitude is None:
            continue

        place_key = next((key for key in known_coordinates if key in site_name.casefold()), None)
        historical_site = site_name
        if place_key == "chennai":
            historical_site = "Tondaimandalam (region; present-day Chennai area)"

        repository = manuscript.get("repository_location") or {}
        if isinstance(repository, dict):
            modern_name = repository.get("location_name") or repository.get("name") or "Archive not recorded"
            modern_lat = repository.get("latitude") or manuscript.get("modern_lat")
            modern_lon = repository.get("longitude") or manuscript.get("modern_lon")
            modern_type = repository.get("location_type") or repository.get("type") or "Repository"
        else:
            modern_name = str(repository) if repository else "Archive not recorded"
            modern_lat = manuscript.get("modern_lat")
            modern_lon = manuscript.get("modern_lon")
            modern_type = "Repository"

        locations.append({
            "id": manuscript["id"], "title": manuscript.get("title", "Untitled manuscript"),
            "location": site_name, "historical_location": historical_site,
            "latitude": float(latitude), "longitude": float(longitude),
            "dynasty": manuscript.get("dynasty", "Unknown"),
            "period": manuscript.get("period", "Period not recorded"),
            "modern_name": modern_name, "modern_type": modern_type,
            "modern_lat": float(modern_lat) if modern_lat is not None else None,
            "modern_lon": float(modern_lon) if modern_lon is not None else None,
        })
    if not locations:
        # Give the historical layer useful, explicitly approximate reference
        # points until the user adds manuscripts with location metadata.
        locations = [
            {
                "id": "reference-thanjavur", "is_reference": True,
                "title": "Thanjavur — Chola-era reference center",
                "location": "Thanjavur", "historical_location": "Thanjavur (Chola-era center)",
                "latitude": 10.7870, "longitude": 79.1378,
                "dynasty": "Chola", "period": "Medieval period",
                "modern_name": "Saraswathi Mahal Library", "modern_type": "Manuscript library",
                "modern_lat": 10.7925, "modern_lon": 79.1350,
            },
            {
                "id": "reference-madurai", "is_reference": True,
                "title": "Madurai — Pandya-era reference center",
                "location": "Madurai", "historical_location": "Madurai (Pandya-era center)",
                "latitude": 9.9252, "longitude": 78.1198,
                "dynasty": "Pandya", "period": "Ancient and medieval periods",
                "modern_name": "Archive location not recorded", "modern_type": "Reference only",
                "modern_lat": None, "modern_lon": None,
            },
            {
                "id": "reference-kanchi", "is_reference": True,
                "title": "Kanchi — Pallava-era reference center",
                "location": "Kanchipuram", "historical_location": "Kanchi (present-day Kanchipuram)",
                "latitude": 12.8342, "longitude": 79.7036,
                "dynasty": "Pallava", "period": "Pallava period",
                "modern_name": "Archive location not recorded", "modern_type": "Reference only",
                "modern_lat": None, "modern_lon": None,
            },
            {
                "id": "reference-tondai", "is_reference": True,
                "title": "Tondaimandalam — historic region reference",
                "location": "Present-day Chennai area", "historical_location": "Tondaimandalam (region; present-day Chennai area)",
                "latitude": 13.0827, "longitude": 80.2707,
                "dynasty": "Pallava / Chola region", "period": "Historic region; periods varied",
                "modern_name": "Archive location not recorded", "modern_type": "Reference only",
                "modern_lat": None, "modern_lon": None,
            },
        ]
    return render_template("map.html", locations=locations)


@app.route("/analytics")
def analytics():
    data = _build_analytics()
    return render_template("analytics.html", data=data)


def _breakdown(records, field, label):
    counts = Counter(str(record.get(field) or "Unknown") for record in records)
    return [{label: name, "count": count} for name, count in sorted(counts.items())]


def _build_analytics():
    """Build transparent catalogue analytics from stored manuscript records."""
    records = get_all_manuscripts()
    artifact_total = sum(len(get_artifacts(record["id"])) for record in records)
    tag_total = sum(len(get_tags(record["id"])) for record in records)
    confidences = [float(record["confidence"]) for record in records if record.get("confidence") is not None]
    confidence_buckets = Counter(
        "0–49%" if value < .5 else "50–69%" if value < .7 else "70–84%" if value < .85 else "85–100%"
        for value in confidences
    )
    location_count = len({_normalise(record.get("location")) for record in records if record.get("location")})
    return {
        "total": len(records),
        "total_images": sum(1 for record in records if record.get("filename")),
        "total_artifacts": artifact_total,
        "total_tags": tag_total,
        "total_locations": location_count,
        "total_scripts": len({record.get("style") for record in records if record.get("style")}),
        # This is intentionally labelled as classification confidence in the UI;
        # OCR confidence should be added when the database stores it.
        "average_classification_confidence": round(sum(confidences) / len(confidences), 3) if confidences else None,
        "by_category": _breakdown(records, "category", "category"),
        "by_dynasty": _breakdown(records, "dynasty", "dynasty"),
        "by_period": _breakdown(records, "period", "period"),
        "by_script": _breakdown(records, "style", "script"),
        "by_location": _breakdown(records, "location", "location"),
        "confidence_distribution": [
            {"range": label, "count": confidence_buckets.get(label, 0)}
            for label in ("0–49%", "50–69%", "70–84%", "85–100%")
        ],
    }


@app.route("/achievements")
def achievements():
    items = get_achievements()
    return render_template("achievements.html", items=items)


@app.route("/compare")
def compare():
    records = get_all_manuscripts()

    def selected_record(value):
        try:
            record_id = int(value)
        except (TypeError, ValueError):
            return None
        return next((record for record in records if record.get("id") == record_id), None)

    left = selected_record(request.args.get("left"))
    right = selected_record(request.args.get("right"))
    if records and left is None:
        left = records[0]
    if len(records) > 1 and (right is None or right.get("id") == left.get("id")):
        right = next(record for record in records if record.get("id") != left.get("id"))

    fields = [
        ("collection", "Collection", lambda record: record.get("collection") or record.get("category")),
        ("dynasty", "Dynasty", lambda record: record.get("dynasty")),
        ("period", "Period", lambda record: record.get("period")),
        ("language", "Language (recorded)", lambda record: record.get("language")),
        ("script_name", "Historical script (recorded)", lambda record: record.get("script_name")),
        ("style", "Classifier label (not a script name)", lambda record: record.get("style")),
        ("location", "Repository / location", lambda record: record.get("location")),
    ]
    comparison = []
    for key, label, get_value in fields:
        left_value = get_value(left) if left else None
        right_value = get_value(right) if right else None
        comparison.append({
            "key": key,
            "label": label,
            "left": left_value or "Not recorded",
            "right": right_value or "Not recorded",
            "same": bool(left_value and right_value and str(left_value).casefold() == str(right_value).casefold()),
        })

    source_urls = {
        "Tamil-Brahmi": "https://www.tnarch.gov.in/tamil-brahmi-script",
        "Tamil script": "https://www.tnarch.gov.in/epigraphy/inscriptions-tamil-script",
        "Grantha": "https://www.tnarch.gov.in/epigraphy/inscriptions-grantha-script",
        "Sinhala language": "https://lrc.columbia.edu/languages/sinhala",
        "Sinhala script": "https://soas-repository.worktribe.com/output/404527/the-evolution-of-the-sinhalese-script-from-the-6th-to-the-10th-century-ad",
    }
    return render_template(
        "compare.html", records=records, left=left, right=right,
        comparison=comparison, source_urls=source_urls,
    )


@app.route("/bulk")
def bulk():
    return redirect(url_for("index") + "#digitize")


# ---------------------------------------------------------------------------
# Export & Comment API Routes
# ---------------------------------------------------------------------------

@app.route("/export/csv")
def export_csv():
    manuscripts = get_all_manuscripts()
    si = StringIO()
    cw = csv.writer(si)
    cw.writerow(["ID", "Title", "Category", "Dynasty", "Period", "Location", "Script Style", "Confidence", "Created At"])
    for m in manuscripts:
        cw.writerow([m["id"], m["title"], m["category"], m["dynasty"], m["period"], m["location"], m["style"], f"{int((m.get('confidence') or 0)*100)}%", m["created_at"]])
    output = si.getvalue()
    return Response(output, mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=manuscripts_export.csv"})


@app.route("/api/manuscript/<int:mid>/comment", methods=["POST"])
def add_manuscript_comment(mid):
    user = session.get("user", {"username": "Anonymous Guest", "role": "Visitor"})
    data = request.json or {}
    text = data.get("comment", "").strip()
    if not text:
        return jsonify({"error": "Comment text cannot be empty"}), 400

    add_comment(mid, user["username"], user["role"], text)
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.route("/api/process", methods=["POST"])
def process():
    """Full pipeline: upload → preprocess → restore → OCR → clean → classify → knowledge → save."""
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    title    = request.form.get("title", "").strip()
    location = request.form.get("location", "").strip() or "Not recorded"
    lat      = request.form.get("latitude", None)
    lon      = request.form.get("longitude", None)

    uploaded_filename = secure_filename(file.filename)
    extension = uploaded_filename.rsplit(".", 1)[1].lower()
    if not title:
        original_stem = os.path.splitext(os.path.basename(file.filename))[0]
        title = " ".join(original_stem.replace("_", " ").replace("-", " ").split()).title()
        if not title:
            title = "Untitled Manuscript"
    original_pdf_url = None
    pdf_page_count = None
    processing_note = None

    if extension == "pdf":
        try:
            import pypdfium2 as pdfium
        except ImportError:
            return jsonify({
                "error": "PDF support needs pypdfium2. Install it with: python -m pip install pypdfium2"
            }), 503

        pdf_bytes = file.read()
        try:
            with PDF_RENDER_LOCK:
                pdf_document = pdfium.PdfDocument(pdf_bytes)
                try:
                    pdf_page_count = len(pdf_document)
                    if pdf_page_count < 1:
                        return jsonify({"error": "This PDF does not contain any pages."}), 400

                    page = pdf_document[0]
                    try:
                        page_width, page_height = page.get_size()
                        page_area = max(page_width * page_height, 1)
                        dpi = min(180, int(72 * (24_000_000 / (page_area * (180 / 72) ** 2)) ** 0.5))
                        dpi = max(72, dpi)
                        filename = f"{os.path.splitext(uploaded_filename)[0]}_page1.png"
                        orig_path = os.path.join(UPLOAD_FOLDER, filename)
                        page.render(scale=dpi / 72).to_pil().convert("RGB").save(orig_path, format="PNG")
                    finally:
                        page.close()
                finally:
                    pdf_document.close()
        except Exception:
            return jsonify({"error": "Could not read this PDF. It may be invalid or password-protected."}), 400

        pdf_path = os.path.join(UPLOAD_FOLDER, uploaded_filename)
        with open(pdf_path, "wb") as pdf_file:
            pdf_file.write(pdf_bytes)
        original_pdf_url = f"/uploads/{uploaded_filename}"
    else:
        filename = uploaded_filename
        orig_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(orig_path)

        # If a user uploads a PalmLeaf-generated intermediate, reuse its
        # original sibling when available instead of thresholding an already
        # processed image again. This also refreshes the normal *_pre / *_res
        # artifacts for that original scan.
        stem, suffix = os.path.splitext(filename)
        source_stem = stem
        derivative_suffixes = ("_retry_res", "_retry_pre", "_pre", "_res")
        removed_suffix = False
        while True:
            match = next((item for item in derivative_suffixes if source_stem.lower().endswith(item)), None)
            if not match:
                break
            source_stem = source_stem[:-len(match)]
            removed_suffix = True
        if removed_suffix:
            candidate_name = source_stem + suffix
            candidate_path = os.path.join(UPLOAD_FOLDER, candidate_name)
            if os.path.isfile(candidate_path):
                filename = candidate_name
                orig_path = candidate_path
                processing_note = (
                    f"The selected file was a PalmLeaf processed image. "
                    f"Analysis used its original scan, {candidate_name}."
                )

    # 1. Load
    original = load_image(orig_path)

    # 2. Preprocess
    preprocessed = preprocess_image(original)

    # 3. Restore (use ML inference module)
    restored = restore_image_ml(preprocessed)

    # 4. Save outputs
    base = os.path.splitext(filename)[0]
    pre_path = os.path.join(UPLOAD_FOLDER, f"{base}_pre.png")
    res_path = os.path.join(UPLOAD_FOLDER, f"{base}_res.png")
    save_image(preprocessed, pre_path)
    save_image(restored, res_path)

    # 5. OCR
    raw_text = extract_text(restored)

    # 6. Text cleaning — use ML corrector if trained, else rule-based
    correction = correct_tamil_ocr(raw_text)
    clean_text = correction["corrected_text"]
    text_diff  = diff_text(raw_text, clean_text)

    # 7. Classification — use ML classifier if trained, else heuristic
    classification = classify_image_ml(original)

    # 8. Knowledge extraction
    knowledge = extract_knowledge(clean_text)

    # 9. Save to DB
    ms_data = {
        "title":         title,
        "filename":      filename,
        "original_path": f"/uploads/{filename}",
        "pre_path":      pre_path,
        "restored_path": res_path,
        "raw_text":      raw_text,
        "clean_text":    clean_text,
        "modern_text":   clean_text,
        "uncertain_words": correction.get("uncertain_words", []),
        "category":      knowledge["category"],
        "dynasty":       knowledge["dynasty"],
        "period":        knowledge["period"],
        "location":      location,
        "style":         classification["style"],
        "confidence":    classification["confidence"],
    }
    if original_pdf_url:
        ms_data["source_pdf_path"] = original_pdf_url
        ms_data["source_pdf_page_count"] = pdf_page_count
    if lat and lon:
        try:
            ms_data["latitude"]  = float(lat)
            ms_data["longitude"] = float(lon)
        except ValueError:
            pass

    ms_id = save_manuscript(ms_data)

    # 10. Save knowledge tags
    if knowledge["tags"]:
        save_tags(ms_id, knowledge["tags"])

    # 11. Auto-achievements
    _check_achievements(ms_id, ms_data, clean_text)

    user_name = session.get("user", {}).get("username", "System / Guest")
    log_action("Manuscript Uploaded", user_name, f"Processed manuscript #{ms_id} ('{title}')")

    base_name = os.path.splitext(filename)[0]

    return jsonify({
        "id":               ms_id,
        "original":         to_base64(original),
        "preprocessed":     to_base64(preprocessed),
        "restored":         to_base64(restored),
        "original_url":     f"/uploads/{filename}",
        "source_pdf_url":   original_pdf_url,
        "source_pdf_page_count": pdf_page_count,
        "processing_note":  processing_note,
        "preprocessed_url": f"/uploads/{base_name}_pre.png",
        "restored_url":     f"/uploads/{base_name}_res.png",
        "raw_text":         raw_text,
        "clean_text":       clean_text,
        "correction":       correction,
        "text_diff":        text_diff,
        "classification":   classification,
        "knowledge":        knowledge,
    })


@app.route("/api/manuscript/<int:mid>")
def api_manuscript(mid):
    m = get_manuscript(mid)
    if not m:
        return jsonify({"error": "Not found"}), 404
    m["artifacts"] = get_artifacts(mid)
    m["tags"]      = get_tags(mid)
    m["comments"]  = get_comments(mid)
    return jsonify(m)


@app.route("/api/search")
def api_search():
    keyword  = request.args.get("q", "")
    category = request.args.get("category", "")
    dynasty  = request.args.get("dynasty", "")
    results  = search_manuscripts(keyword, category, dynasty)
    return jsonify(results)


@app.route("/api/analytics")
def api_analytics():
    return jsonify(_build_analytics())


@app.route("/api/manuscript/<int:mid>/mindmap")
def manuscript_mindmap(mid):
    """Return the connected historical context graph for one manuscript."""
    manuscript = get_manuscript(mid)
    if not manuscript:
        return jsonify({"error": "Manuscript not found"}), 404

    nodes = [{"id": "manuscript", "label": manuscript.get("title", "Manuscript"), "type": "manuscript"}]
    links = []
    used_ids = {"manuscript"}

    def add_node(kind, label):
        label = str(label or "Unknown")
        node_id = f"{kind}-{len(nodes)}"
        if node_id in used_ids:
            return
        used_ids.add(node_id)
        nodes.append({"id": node_id, "label": label, "type": kind})
        links.append({"source": "manuscript", "target": node_id})

    for kind, field in (
        ("dynasty", "dynasty"), ("period", "period"), ("script", "style"),
        ("subject", "category"), ("location", "location")
    ):
        value = manuscript.get(field)
        if value and value != "Unknown":
            add_node(kind, value)

    for tag in get_tags(mid):
        add_node("tag", tag.get("tag"))
    for artifact in get_artifacts(mid):
        add_node("artifact", artifact.get("name"))

    return jsonify({"nodes": nodes, "links": links})


@app.route("/api/model-status")
def api_model_status():
    return jsonify(get_model_status())


@app.route("/api/modernize", methods=["POST"])
def api_modernize_tamil():
    """Convert user-edited or OCR-derived Tamil without blocking damaged text."""
    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "")
    if not isinstance(text, str):
        return jsonify({"error": "Tamil text must be a string."}), 400
    if len(text) > 12000:
        return jsonify({"error": "Please convert passages of 12,000 characters or fewer."}), 413
    correction = correct_tamil_ocr(text)
    result = modernize_tamil(text, correction["corrected_text"], correction)
    return jsonify(result)


@app.route("/api/retry-ocr", methods=["POST"])
def api_retry_ocr():
    """Rerun preprocessing, restoration, and Tesseract for a saved manuscript."""
    payload = request.get_json(silent=True) or {}
    try:
        manuscript_id = int(payload.get("id", 0))
    except (TypeError, ValueError):
        manuscript_id = 0
    manuscript = get_manuscript(manuscript_id) if manuscript_id else None
    if not manuscript:
        return jsonify({"error": "Save or open a manuscript before retrying OCR."}), 404

    filename = os.path.basename(str(manuscript.get("filename", "")))
    if not filename:
        return jsonify({"error": "The original scan is not available for OCR retry."}), 404
    if filename.lower().endswith(".pdf"):
        filename = f"{os.path.splitext(filename)[0]}_page1.png"
    source_path = os.path.abspath(os.path.join(UPLOAD_FOLDER, filename))
    if os.path.commonpath([source_path, os.path.abspath(UPLOAD_FOLDER)]) != os.path.abspath(UPLOAD_FOLDER):
        return jsonify({"error": "Invalid scan path."}), 400
    if not os.path.isfile(source_path):
        return jsonify({"error": "The original scan is no longer available. Upload it again to retry OCR."}), 404

    try:
        original = load_image(source_path)
        restored = restore_image_ml(preprocess_image(original))
        raw_text = extract_text(restored, retry=True)
        correction = correct_tamil_ocr(raw_text)
        clean_text = correction["corrected_text"]
        save_image(restored, os.path.join(UPLOAD_FOLDER, f"{os.path.splitext(filename)[0]}_retry_res.png"))
        update_manuscript(manuscript_id, {
            "raw_text": raw_text,
            "clean_text": clean_text,
            "modern_text": clean_text,
            "uncertain_words": correction.get("uncertain_words", []),
        })
        return jsonify({
            "raw_text": raw_text,
            "correction": correction,
            "clean_text": clean_text,
        })
    except Exception as exc:
        app.logger.exception("OCR retry failed for manuscript %s", manuscript_id)
        return jsonify({"error": f"OCR retry failed: {exc}"}), 500


@app.route("/api/locations")
def api_locations():
    return jsonify(get_locations())


@app.route("/api/artifact/add", methods=["POST"])
def add_artifact():
    data = request.json
    save_artifact(
        data["manuscript_id"],
        data.get("name", ""),
        data.get("description", ""),
        data.get("symbol", "")
    )
    return jsonify({"status": "ok"})


@app.route("/api/manuscript/<int:mid>/update", methods=["POST"])
def update_meta(mid):
    data = request.json
    allowed = ["title", "location", "latitude", "longitude", "dynasty", "category"]
    update_data = {k: v for k, v in data.items() if k in allowed}
    update_manuscript(mid, update_data)
    return jsonify({"status": "ok"})


@app.route("/api/manuscript/<int:mid>/delete", methods=["POST"])
def delete_ms(mid):
    user_name = session.get("user", {}).get("username", "Guest")
    delete_manuscript(mid)
    log_action("Manuscript Deleted", user_name, f"Deleted manuscript #{mid}")
    return jsonify({"status": "deleted"})


@app.route("/manuscript/<int:mid>/export/html")
def export_html(mid):
    m = get_manuscript(mid)
    if not m:
        return render_template("404.html"), 404
    tags      = get_tags(mid)
    artifacts = get_artifacts(mid)
    html = build_html_report(m, tags, artifacts)
    return Response(html, mimetype="text/html",
                    headers={"Content-Disposition": f'attachment; filename="manuscript_{mid}.html"'})


@app.route("/manuscript/<int:mid>/export/txt")
def export_txt(mid):
    m = get_manuscript(mid)
    if not m:
        return render_template("404.html"), 404
    tags = get_tags(mid)
    txt  = build_text_report(m, tags)
    return Response(txt, mimetype="text/plain",
                    headers={"Content-Disposition": f'attachment; filename="manuscript_{mid}.txt"'})


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


# ---------------------------------------------------------------------------
# Achievement logic
# ---------------------------------------------------------------------------

def _check_achievements(ms_id, data, text):
    all_ms = get_all_manuscripts()

    if len(all_ms) == 1:
        save_achievement(ms_id, "🏆 Pioneer", "First manuscript digitized in this repository!")

    if data.get("dynasty") != "Unknown":
        save_achievement(ms_id, "🏺 Dynasty Identified",
                         f"Dynasty detected: {data['dynasty']}")

    if len(text) > 500:
        save_achievement(ms_id, "📜 Rich Text",
                         "Manuscript contains more than 500 characters of Tamil text.")

    if data.get("category") == "Medicine":
        save_achievement(ms_id, "🌿 Medical Knowledge",
                         "Manuscript contains ancient medical knowledge.")


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 32 MB."}), 413


if __name__ == "__main__":
    app.run(debug=True, port=5000)
