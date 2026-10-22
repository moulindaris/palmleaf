"""MongoDB persistence layer for Palmleaf."""
from datetime import datetime, timezone

from pymongo import ASCENDING, DESCENDING, MongoClient, TEXT, ReturnDocument
from pymongo.errors import DuplicateKeyError
from werkzeug.security import check_password_hash, generate_password_hash

from config import Config

client = MongoClient(Config.MONGO_URI, serverSelectionTimeoutMS=5000)
db = client.get_default_database()


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _plain(document):
    if document is None:
        return None
    result = dict(document)
    result.pop("_id", None)
    return result


def init_db():
    """Check the MongoDB connection and initialize indexes."""
    client.admin.command("ping")
    db.manuscripts.create_index([("id", ASCENDING)], unique=True)
    db.manuscripts.create_index([
        ("title", TEXT), ("raw_ocr", TEXT), ("cleaned_text", TEXT),
        ("raw_text", TEXT), ("clean_text", TEXT),
    ])
    db.manuscripts.create_index([
        ("dynasty", ASCENDING), ("period", ASCENDING), ("category", ASCENDING)
    ])
    db.locations.create_index([("location_name", ASCENDING), ("location_type", ASCENDING)])
    db.tdl_catalog.create_index([("source_id", ASCENDING)], unique=True)
    db.tdl_catalog.create_index([("title", TEXT), ("publisher", TEXT), ("subject", TEXT)])
    db.cict_catalog.create_index([("source_id", ASCENDING)], unique=True)
    for collection in (db.artifacts, db.tags, db.comments, db.achievements):
        collection.create_index([("manuscript_id", ASCENDING)])
    db.users.create_index([("username", ASCENDING)], unique=True)
    db.users.create_index([("email", ASCENDING)], unique=True)
    db.system_logs.create_index([("created_at", DESCENDING)])
    print(f"MongoDB connected: {db.name}")


def _next_id(sequence):
    counter = db.counters.find_one_and_update(
        {"_id": sequence}, {"$inc": {"value": 1}},
        upsert=True, return_document=ReturnDocument.AFTER,
    )
    return counter["value"]


def save_manuscript(data):
    record = dict(data)
    record["id"] = _next_id("manuscript")
    record["created_at"] = _now()
    db.manuscripts.insert_one(record)
    return record["id"]


def update_manuscript(manuscript_id, data):
    db.manuscripts.update_one({"id": int(manuscript_id)}, {"$set": dict(data)})


def get_manuscript(manuscript_id):
    return _plain(db.manuscripts.find_one({"id": int(manuscript_id)}))


def get_all_manuscripts():
    return [_plain(item) for item in db.manuscripts.find().sort("id", DESCENDING)]


def get_tdl_catalog_records():
    """Return sourced catalog metadata imported from Tamil Digital Library."""
    return [_plain(item) for item in db.tdl_catalog.find().sort("title", ASCENDING)]


def get_cict_catalog_records():
    """Return work-level CICT archive listings and their source links."""
    return [_plain(item) for item in db.cict_catalog.find().sort("title", ASCENDING)]


def search_manuscripts(keyword="", category="", dynasty=""):
    query = {}
    if category:
        query["category"] = {"$regex": category, "$options": "i"}
    if dynasty:
        query["dynasty"] = {"$regex": dynasty, "$options": "i"}
    if keyword:
        query["$text"] = {"$search": keyword}
    return [_plain(item) for item in db.manuscripts.find(query).sort("id", DESCENDING)]


def delete_manuscript(manuscript_id):
    mid = int(manuscript_id)
    db.manuscripts.delete_one({"id": mid})
    for collection in (db.tags, db.artifacts, db.comments, db.achievements):
        collection.delete_many({"manuscript_id": mid})


def save_tags(manuscript_id, tags):
    documents = []
    for tag in tags:
        if isinstance(tag, dict):
            item = dict(tag)
        elif isinstance(tag, (tuple, list)):
            item = {"tag": str(tag[0]), "category": str(tag[1]) if len(tag) > 1 else ""}
        else:
            item = {"tag": str(tag), "category": ""}
        item["manuscript_id"] = int(manuscript_id)
        documents.append(item)
    if documents:
        db.tags.insert_many(documents)


def get_tags(manuscript_id):
    return [_plain(item) for item in db.tags.find({"manuscript_id": int(manuscript_id)})]


def save_artifact(manuscript_id, name, description="", symbol=""):
    db.artifacts.insert_one({
        "manuscript_id": int(manuscript_id), "name": name,
        "description": description, "symbol": symbol, "created_at": _now(),
    })


def get_artifacts(manuscript_id):
    return [_plain(item) for item in db.artifacts.find({"manuscript_id": int(manuscript_id)})]


def save_achievement(manuscript_id, name, description=""):
    db.achievements.insert_one({
        "manuscript_id": int(manuscript_id), "title": name,
        "description": description, "awarded_at": _now(),
    })


def get_achievements():
    items = []
    for item in db.achievements.find().sort("awarded_at", DESCENDING):
        record = _plain(item)
        manuscript = db.manuscripts.find_one({"id": record["manuscript_id"]}, {"title": 1})
        record["manuscript_title"] = manuscript.get("title", "Deleted manuscript") if manuscript else "Deleted manuscript"
        items.append(record)
    return items


def get_locations():
    locations = [_plain(item) for item in db.locations.find()]
    known = {item.get("location_name") for item in locations}
    for name in db.manuscripts.distinct("location"):
        if name and name not in known:
            locations.append({"location_name": name, "location_type": "Manuscript location"})
    return locations


def get_analytics():
    records = get_all_manuscripts()
    return {"total": len(records), "by_category": _counts(records, "category"),
            "by_dynasty": _counts(records, "dynasty")}


def _counts(records, field):
    counts = {}
    for record in records:
        value = record.get(field) or "Unknown"
        counts[value] = counts.get(value, 0) + 1
    return [{field: name, "count": count} for name, count in sorted(counts.items())]


def create_user(username, email, password, role="Student / Researcher"):
    try:
        db.users.insert_one({
            "id": _next_id("user"), "username": username, "email": email,
            "password_hash": generate_password_hash(password), "role": role,
            "created_at": _now(),
        })
        return True
    except DuplicateKeyError:
        return False


def verify_user(username, password):
    user = db.users.find_one({"username": username})
    if not user or not check_password_hash(user.get("password_hash", ""), password):
        return None
    return {key: user[key] for key in ("id", "username", "email", "role")}


def get_all_users():
    return [{key: item.get(key) for key in ("id", "username", "email", "role", "created_at")}
            for item in db.users.find().sort("id", DESCENDING)]


def add_comment(manuscript_id, user_name, role, text):
    db.comments.insert_one({
        "manuscript_id": int(manuscript_id), "user_name": user_name,
        "role": role, "text": text, "created_at": _now(),
    })


def get_comments(manuscript_id):
    return [_plain(item) for item in db.comments.find(
        {"manuscript_id": int(manuscript_id)}).sort("created_at", ASCENDING)]


def log_action(action, user_name, details=""):
    timestamp = _now()
    db.system_logs.insert_one({
        "action": action, "user_name": user_name,
        "details": details, "timestamp": timestamp, "created_at": timestamp,
    })


def get_system_logs(limit=100):
    return [_plain(item) for item in db.system_logs.find().sort("created_at", DESCENDING).limit(int(limit))]


def get_conn():
    """Return the MongoDB database for legacy ingestion scripts."""
    return db
