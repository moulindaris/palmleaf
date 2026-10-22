"""Create a few clearly labeled, source-linked catalog examples for demos.

These records demonstrate the repository UI; they are not digitized folios and
do not include manuscript images, OCR text, or model predictions.
"""
from datetime import datetime, timezone

from database.db import _next_id, db, init_db


SAMPLE_TITLES = {
    "Tirukkural": "A classical Tamil work listed in CICT's Pathinenkilkanakku collection.",
    "Tolkappiyam": "A classical Tamil grammar work listed in CICT's Grammar collection.",
    "Silappathikaram": "A classical Tamil epic listed in CICT's Epics collection.",
}
SOURCE_NAME = "Central Institute of Classical Tamil (CICT) Digital Archive"


def seed():
    init_db()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for title, description in SAMPLE_TITLES.items():
        source = db.cict_catalog.find_one({"title": title})
        if not source:
            print(f"Skipped {title}: no matching CICT catalog record found")
            continue

        source_id = source["source_id"]
        existing = db.manuscripts.find_one({"source_id": source_id})
        record_id = existing["id"] if existing else _next_id("manuscript")
        db.manuscripts.update_one(
            {"source_id": source_id},
            {
                "$set": {
                    "id": record_id,
                    "title": title,
                    "category": source["collection"],
                    "collection": source["collection"],
                    "description": description,
                    "location": "CICT Digital Archive (online catalog)",
                    "dynasty": "Not specified in source listing",
                    "period": "Not specified in source listing",
                    "source_name": SOURCE_NAME,
                    "source_id": source_id,
                    "source_url": source["source_url"],
                    "source_institution_count": source["source_institution_count"],
                    "pothi_bundle_count": source["pothi_bundle_count"],
                    "record_type": "work_level_catalog_listing",
                    "demo_record": True,
                    "style": "",
                    "confidence": None,
                },
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
        )
    total = db.manuscripts.count_documents({"demo_record": True})
    print(f"CICT catalog examples ready: {total} in {db.name}.manuscripts")


if __name__ == "__main__":
    seed()
