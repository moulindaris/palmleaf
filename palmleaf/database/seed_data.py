from database.db import db, init_db

def seed():
    init_db()
    db.manuscripts.drop()
    db.locations.drop()
    db.symbols.drop()

    locations = [
        {
            "location_id": "LOC-001",
            "location_name": "Thanjavur Brihadisvara Complex",
            "location_type": "Discovery Site (Ancient)",
            "latitude": 10.7828, "longitude": 79.1318,
            "region": "Chola Nadu", "country": "India",
            "dynasties_associated": ["Chola"]
        },
        {
            "location_id": "LOC-002",
            "location_name": "Saraswathi Mahal Library",
            "location_type": "Modern Repository",
            "latitude": 10.7925, "longitude": 79.1350,
            "region": "Thanjavur", "country": "India",
            "dynasties_associated": ["Maratha", "Nayaka", "Chola"]
        },
        {
            "location_id": "LOC-003",
            "location_name": "Keezhadi Archaeological Excavation",
            "location_type": "Discovery Site (Ancient)",
            "latitude": 9.8631, "longitude": 78.1887,
            "region": "Pandya Nadu", "country": "India",
            "dynasties_associated": ["Pandya", "Sangam Era"]
        }
    ]
    db.locations.insert_many(locations)

    db.manuscripts.insert_one({
        "manuscript_id": "TM-CHOLA-001",
        "title": "Siddha Herbal Pharmacology (சித்த மருத்துவச் சுவடி)",
        "language": "Tamil",
        "estimated_date": "1045 CE",
        "condition": "Brittle, partial edge fracture, high contrast loss",
        "dynasty": "Chola",
        "period": "Medieval Chola Empire",
        "author": "Agasthiyar Lineage Practitioner",
        "subject": "Medicine & Alchemy",
        "script": "Tamil-Grantha Hybrid",
        "original_image_path": "/static/uploads/sample_leaf.jpg",
        "restored_image_path": "/static/restored/sample_leaf_restored.png",
        "raw_ocr": "நமசிவாய வாழ்க நாதன் தாள் வாழ்க மூலிகை முறை வாதம் பித்தம் நீக்கும் முறைமை...",
        "ocr_confidence": 0.892,
        "cleaned_text": "நமசிவாய வாழ்க நாதன் தாள் வாழ்க. மூலிகை முறை வாதம், பித்தம், கபம் தீர்க்கும் உன்னத முறைமை ஆகும்.",
        "keywords": ["மருத்துவம்", "மூலிகை", "வாதம்", "பித்தம்"],
        "discovery_location": locations[0],
        "repository_location": locations[1],
        "artifacts": [{"artifact_id": "ART-01", "name": "Copper Palm Stylus (எழுத்தாணி)", "type": "Tool", "era": "11th Century"}],
        "symbols": [{"symbol_id": "SYM-01", "name": "Pillaiyar Suzhi (௳)", "meaning": "Auspicious invocation begin marker"}],
        "references": [{"title": "Epigraphia Carnatica Vol. V", "type": "Archaeological Report", "year": 1902, "doi": "10.1000/182"}]
    })
    print("Sample Manuscript and Historical Locations Seeded.")

if __name__ == "__main__":
    seed()
