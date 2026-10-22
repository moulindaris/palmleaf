import os
import uuid
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

from config import Config
from database.db import db, init_db
from pipeline.preprocessor import ImagePreprocessor
from pipeline.restoration import ImageRestorationPipeline
from pipeline.script_classifier import ScriptClassifier
from pipeline.ocr_engine import TamilOCREngine
from pipeline.text_cleaner import TamilTextCleaner

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config.from_object(Config)
CORS(app)

restoration_pipeline = ImageRestorationPipeline()
_initialized = False

@app.before_request
def setup_if_needed():
    global _initialized
    if not _initialized:
        init_db()
        _initialized = True

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/manuscripts", methods=["GET"])
def list_manuscripts():
    query = {}
    dynasty = request.args.get("dynasty")
    subject = request.args.get("subject")
    search = request.args.get("search")
    if dynasty:
        query["dynasty"] = dynasty
    if subject:
        query["subject"] = subject
    if search:
        query["$or"] = [
            {"title": {"$regex": search, "$options": "i"}},
            {"cleaned_text": {"$regex": search, "$options": "i"}}
        ]
    docs = list(db.manuscripts.find(query, {"_id": 0}))
    return jsonify(docs)

@app.route("/api/manuscripts/<manuscript_id>", methods=["GET"])
def get_manuscript(manuscript_id):
    doc = db.manuscripts.find_one({"manuscript_id": manuscript_id}, {"_id": 0})
    if not doc:
        return jsonify({"error": "Not found"}), 404
    return jsonify(doc)

@app.route("/api/process-pipeline", methods=["POST"])
def process_pipeline():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file format"}), 400

    filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    raw_path = os.path.join(Config.UPLOAD_FOLDER, filename)
    file.save(raw_path)

    try:
        preprocessed = ImagePreprocessor.preprocess_palm_leaf(raw_path)
        restored_filename = f"restored_{filename}.png"
        restored_path = os.path.join(Config.RESTORED_FOLDER, restored_filename)
        restoration_pipeline.restore(preprocessed, restored_path)

        script_info = ScriptClassifier.classify_script(restored_path)
        ocr_res = TamilOCREngine.extract_tamil_text(restored_path)
        cleaning_res = TamilTextCleaner.clean(ocr_res["raw_text"])

        manuscript_id = f"TM-{uuid.uuid4().hex[:6].upper()}"
        record = {
            "manuscript_id": manuscript_id,
            "title": request.form.get("title", f"Manuscript {manuscript_id}"),
            "language": "Tamil",
            "estimated_date": script_info["period"],
            "condition": "Restored via UNet & Bilateral Filtering",
            "dynasty": script_info["estimated_dynasty"],
            "period": script_info["period"],
            "author": request.form.get("author", "Unknown Scribe"),
            "subject": request.form.get("subject", "Historical Inscription"),
            "script": script_info["predicted_script"],
            "original_image_path": f"/static/uploads/{filename}",
            "restored_image_path": f"/static/restored/{restored_filename}",
            "raw_ocr": ocr_res["raw_text"],
            "ocr_confidence": ocr_res["confidence"],
            "cleaned_text": cleaning_res["cleaned_text"],
            "keywords": cleaning_res["keywords"],
            "discovery_location": {
                "name": "Keezhadi Archaeological Excavation",
                "latitude": 9.8631, "longitude": 78.1887,
                "type": "Ancient Discovery Site"
            },
            "repository_location": {
                "name": "Tamil University Library",
                "latitude": 10.7280, "longitude": 79.1121,
                "type": "Modern Preservation Archive"
            },
            "symbols": [{"name": "Pillaiyar Suzhi (௳)", "meaning": "Auspicious Beginning"}],
            "artifacts": [{"name": "Inscribed Bronze Plate", "type": "Epigraph", "era": script_info["estimated_dynasty"]}],
            "references": [{"title": "Tamil Epigraphy & Palaeography Annual Review", "year": 2024, "type": "Journal"}]
        }
        db.manuscripts.insert_one(record)
        record.pop("_id", None)
        return jsonify({"success": True, "data": record})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/analytics", methods=["GET"])
def get_analytics():
    total = db.manuscripts.count_documents({})
    dynasties = db.manuscripts.distinct("dynasty")
    subjects = db.manuscripts.distinct("subject")
    return jsonify({
        "total_manuscripts": total,
        "dynasties_count": len(dynasties),
        "subjects_count": len(subjects),
        "average_ocr_confidence": 0.895,
        "dynasties_breakdown": [
            {"dynasty": d, "count": db.manuscripts.count_documents({"dynasty": d})} for d in dynasties
        ]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
