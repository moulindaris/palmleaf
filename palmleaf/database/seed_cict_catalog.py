"""Import work-level listings from CICT's Digital Library of Tamil Palm-Leaf Manuscripts.

This stores titles, collection groupings, source/pothi counts, and links only.
It does not copy manuscript images, transcriptions, or full texts.
"""
from datetime import datetime, timezone

from database.db import db, init_db


SOURCE = "Central Institute of Classical Tamil (CICT) Digital Archive"
ARCHIVE = "https://www.digitalarchives.cict.in/"

# Titles and collection counts transcribed from the CICT archive's Browse the
# Manuscript Collection section. These are work-level listings, not 41 scans.
WORKS = [
    ("Tolkappiyam", "Grammar", 9, 212),
    ("Iraiyanar Akapporul", "Grammar", 7, 17),
    ("Natrinai", "Ettuttokai", 2, 10),
    ("Kuruntokai", "Ettuttokai", 3, 15),
    ("Ainkurunuru", "Ettuttokai", 3, 8),
    ("Pathirruppathu", "Ettuttokai", 4, 8),
    ("Paripatal", "Ettuttokai", 3, 5),
    ("Kalittokai", "Ettuttokai", 2, 10),
    ("Akananuru", "Ettuttokai", 4, 24),
    ("Purananuru", "Ettuttokai", 6, 24),
    ("Tirumurukarruppatai", "Pattuppattu", 6, 48),
    ("Porunararruppatai", "Pattuppattu", 2, 9),
    ("Cirupanarruppatai", "Pattuppattu", 2, 9),
    ("Perumpanarruppatai", "Pattuppattu", 2, 10),
    ("Mullaippattu", "Pattuppattu", 2, 4),
    ("Maduraikkanchi", "Pattuppattu", 2, 5),
    ("Nedunalvadai", "Pattuppattu", 2, 4),
    ("Kurinjippattu", "Pattuppattu", 2, 3),
    ("Pattinappalai", "Pattuppattu", 2, 3),
    ("Malaipadukadam", "Pattuppattu", 2, 3),
    ("Naladiyar", "Pathinenkilkanakku", 10, 76),
    ("Nanmanikkadigai", "Pathinenkilkanakku", 5, 15),
    ("Inna Narpathu", "Pathinenkilkanakku", 4, 17),
    ("Iniyavai Narpathu", "Pathinenkilkanakku", 4, 18),
    ("Kar Narpathu", "Pathinenkilkanakku", 3, 14),
    ("Kalavali Narpathu", "Pathinenkilkanakku", 3, 13),
    ("Aintinai Aimpathu", "Pathinenkilkanakku", 3, 13),
    ("Aintinai Elupathu", "Pathinenkilkanakku", 3, 6),
    ("Thinai Mozhi Aimpathu", "Pathinenkilkanakku", 3, 8),
    ("Thinai Malai Nurru Aimpathu", "Pathinenkilkanakku", 2, 8),
    ("Pazhamozhi Nanuru", "Pathinenkilkanakku", 4, 14),
    ("Sirupanchamulam", "Pathinenkilkanakku", 2, 10),
    ("Tirukkural", "Pathinenkilkanakku", 12, 135),
    ("Tirikatukam", "Pathinenkilkanakku", 5, 21),
    ("Acharakkovai", "Pathinenkilkanakku", 5, 15),
    ("Muthumozhikkanchi", "Pathinenkilkanakku", 3, 8),
    ("Elathi", "Pathinenkilkanakku", 2, 8),
    ("Kainnilai", "Pathinenkilkanakku", 1, 4),
    ("Silappathikaram", "Epics", 4, 17),
    ("Manimekalai", "Epics", 4, 12),
    ("Muttollayiram", "Epics", 1, 3),
]


def seed():
    init_db()
    imported_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for title, collection, source_count, pothi_count in WORKS:
        slug = title.lower().replace(" ", "-")
        source_id = f"cict-work-{slug}"
        is_tirukkural = title == "Tirukkural"
        db.cict_catalog.update_one(
            {"source_id": source_id},
            {"$set": {
                "source_id": source_id,
                "record_type": "literary_work",
                "title": title,
                "subject": "Tamil literature",
                "collection": collection,
                "source_institution_count": source_count,
                "pothi_bundle_count": pothi_count,
                "catalogue_status": "Full metadata shown" if is_tirukkural else "Work listing",
                "source_name": SOURCE,
                "source_url": ARCHIVE + ("#ground-truth" if is_tirukkural else "#browse"),
                "imported_at": imported_at,
            }},
            upsert=True,
        )
    print(f"Imported {len(WORKS)} CICT work listings into {db.name}.cict_catalog")


if __name__ == "__main__":
    seed()
