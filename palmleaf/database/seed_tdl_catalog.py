"""Import verified Tamil Digital Library catalog publications into MongoDB.

These are catalog records with source links, not OCR texts or processed scans.
The importer is idempotent and only updates records carrying its source IDs.
"""
from datetime import datetime, timezone

from database.db import db, init_db


SOURCE = "Tamil Digital Library (Tamil Virtual Academy)"
RECORDS = [
    {
        "source_id": "TVA_BOK_0028191",
        "title": "A descriptive catalogue of Tamil palm-leaf manuscripts",
        "publisher": "Department of Manuscriptology, Institute of Asian Studies, Madras",
        "subject": "Catalogs",
        "holding_institution": "Saraswathi Mahal Library",
        "source_url": "https://tamildigitallibrary.in/Articles/%E0%AE%A8%E0%AF%82%E0%AE%B2%E0%AF%8D-28191-A%20descriptive%20catalogue%20of%20Tamil%20palm-leaf%20manuscripts",
    },
    {
        "source_id": "tdl-descriptive-tamil-palm-leaf-1991",
        "title": "A descriptive catalogue of palm leaf manuscripts in Tamil",
        "publisher": "Institute of Asian Studies, Madras",
        "publication_year": 1991,
        "page_count": 192,
        "subject": "Catalogs",
        "source_url": "https://www.tamildigitallibrary.in/book-detail.php?id=jZY9lup2kZl6TuXGlZQdjZp8jZI8",
    },
    {
        "source_id": "tdl-descriptive-tamil-palm-leaf-1996",
        "title": "A descriptive catalogue of palm-leaf manuscripts in tamil",
        "publisher": "Institute of Asian Studies, Madras",
        "publication_year": 1996,
        "page_count": 473,
        "series": "Publication no. 35",
        "subject": "Catalogs",
        "holding_institution": "International Institute of Tamil Studies",
        "source_url": "https://www.tamildigitallibrary.in/book-detail.php?id=jZY9lup2kZl6TuXGlZQdjZtdkuM7",
    },
    {
        "source_id": "TVA_BOK_0020993",
        "title": "International catalogue of Tamil palmleaf manuscripts, Volume 1 (1–6019)",
        "publisher": "Tamil University, Thanjavur",
        "publication_year": 1991,
        "page_count": 510,
        "edition": "First edition",
        "series": "Tamil University Publication 122-1",
        "subject": "Catalogs",
        "source_url": "https://tamildigitallibrary.in/book-details/TWpBNU9UTT1BdG9aYXRvejB0bzk-/TkE9PUF0b1phdG96MHRvOQ--/TWpBPUF0b1phdG96MHRvOQ--",
    },
    {
        "source_id": "TVA_BOK_0020997",
        "title": "International catalogue of Tamil palmleaf manuscripts, Volume 2 (6020–11183)",
        "publisher": "Tamil University, Thanjavur",
        "publication_year": 1991,
        "series": "Tamil University Publication 122-2",
        "subject": "Catalogs",
        "source_url": "https://tamildigitallibrary.in/Articles/%E0%AE%A8%E0%AF%82%E0%AE%B2%E0%AF%8D-20997-International%20catalogue%20of%20tamil%20palmleaf%20manuscripts",
    },
    {
        "source_id": "TVA_BOK_0021006",
        "title": "International catalogue of Tamil palmleaf manuscripts, Volume 5",
        "publisher": "Tamil University, Thanjavur",
        "publication_year": 1991,
        "page_count": 313,
        "series": "Tamil University Publication 122-5",
        "subject": "Catalogs",
        "source_url": "https://tamildigitallibrary.in/Articles/%E0%AE%A8%E0%AF%82%E0%AE%B2%E0%AF%8D-21006-International%20catalogue%20of%20tamil%20palmleaf%20manuscripts",
    },
    {
        "source_id": "TVA_BOK_0011478",
        "title": "Computerised International Catalogue of Tamil Palmleaf Manuscripts, Volume 5",
        "publisher": "Tamil University, Thanjavur",
        "publication_year": 1989,
        "series": "Tamil University Publication 122-5",
        "subject": "Catalogs",
        "source_url": "https://tamildigitallibrary.in/Articles/%E0%AE%A8%E0%AF%82%E0%AE%B2%E0%AF%8D-11478-Computerised%20International%20Catalogue%20of%20Tamil%20Palmleaf%20Manuscripts",
    },
    {
        "source_id": "TVA_BOK_0021009",
        "title": "An alphabetical index of Sanskrit, Telugu and Tamil manuscripts (palm-leaf and paper)",
        "publisher": "Sri Venkateswara Oriental Research Institute Library, Tirupati",
        "publication_year": 1956,
        "subject": "Manuscripts; Catalogs",
        "source_url": "https://tamildigitallibrary.in/Articles/%E0%AE%A8%E0%AF%82%E0%AE%B2%E0%AF%8D-21009-An%20alphabetical%20index%20of%20Sanskrit%2C%20Telugu%20and%20Tamil%20Manuscripts-%28Palm-Leaf%20and%20paper%29",
    },
    {
        "source_id": "TVA_BOK_0021017",
        "title": "Catalogue of Tamil palmleaf manuscripts in the Tamil University",
        "publisher": "Tamil University, Thanjavur",
        "publication_year": 1985,
        "subject": "Catalogs",
        "source_url": "https://tamildigitallibrary.in/Articles/%E0%AE%A8%E0%AF%82%E0%AE%B2%E0%AF%8D-21017-Catalogue%20of%20tamil%20palmleaf%20manuscripts%20in%20the%20tamil%20university",
    },
    {
        "source_id": "TVA_BOK_0006870",
        "record_type": "literary_work",
        "title": "திருக்குறள்",
        "publisher": "International Institute of Tamil Studies, Chennai",
        "publication_year": 1999,
        "subject": "Tamil literature",
        "keywords": ["அறத்துப்பால்", "பொருட்பால்", "காமத்துப்பால்"],
        "holding_institution": "International Institute of Tamil Studies",
        "source_url": "https://tamildigitallibrary.in/Articles/%E0%AE%A8%E0%AF%82%E0%AE%B2%E0%AF%8D-6870-%E0%AE%A4%E0%AE%BF%E0%AE%B0%E0%AF%81%E0%AE%95%E0%AF%8D%E0%AE%95%E0%AF%81%E0%AE%B1%E0%AE%B3%E0%AF%8D",
    },
    {
        "source_id": "TVA_BOK_0007856",
        "record_type": "literary_work",
        "title": "மணிமேகலை",
        "author": "மதுரைக் கூலவாணிகன் சாத்தனார்",
        "publisher": "Madras Rippon Press, Chennai",
        "publication_year": 1894,
        "subject": "Tamil literature",
        "holding_institution": "Connemara Public Library",
        "source_url": "https://tamildigitallibrary.in/Articles/%E0%AE%A8%E0%AF%82%E0%AE%B2%E0%AF%8D-7856-%E0%AE%AE%E0%AE%A3%E0%AE%BF%E0%AE%AE%E0%AF%87%E0%AE%95%E0%AE%B2%E0%AF%88",
    },
    {
        "source_id": "TVA_BOK_0034515",
        "record_type": "literary_work",
        "title": "மணிமேகலை",
        "author": "ஐயர், ஏ. எஸ். பி.",
        "publisher": "Alliance Company, Chennai",
        "publication_year": 1951,
        "subject": "Tamil literature",
        "keywords": ["கோவலன்", "கண்ணகி", "சிலப்பதிகாரம்"],
        "holding_institution": "Connemara Public Library",
        "source_url": "https://tamildigitallibrary.in/Articles/%E0%AE%A8%E0%AF%82%E0%AE%B2%E0%AF%8D-34515-%E0%AE%AE%E0%AE%A3%E0%AE%BF%E0%AE%AE%E0%AF%87%E0%AE%95%E0%AE%B2%E0%AF%88",
    },
    {
        "source_id": "tdl-book-jZY9lup2kZl6TuXGlZQdjZM3juUy",
        "record_type": "literary_work",
        "title": "சிலப்பதிகாரம்",
        "author": "இளங்கோவடிகள்",
        "publication_year": 1872,
        "subject": "Tamil literature",
        "keywords": ["காப்பியம்", "ஐம்பெருங்காப்பியம்"],
        "holding_institution": "Connemara Public Library",
        "source_url": "https://www.tamildigitallibrary.in/book-detail.php?id=jZY9lup2kZl6TuXGlZQdjZM3juUy",
    },
    {
        "source_id": "tdl-article-44453",
        "record_type": "literary_work",
        "title": "புறநானூறு : 1–200 பாட்டுக்கள்",
        "publisher": "South India Saiva Siddhanta Works Publishing Society, Chennai",
        "publication_year": 1947,
        "subject": "Sangam literature",
        "keywords": ["புறம்", "பாட்டுக்கள்"],
        "source_url": "https://tamildigitallibrary.in/Articles/%E0%AE%A8%E0%AF%82%E0%AE%B2%E0%AF%8D-44453-%E0%AE%AA%E0%AF%81%E0%AE%B1%E0%AE%A8%E0%AE%BE%E0%AE%A9%E0%AF%82%E0%AE%B1%E0%AF%81-1-200%20%E0%AE%AA%E0%AE%BE%E0%AE%9F%E0%AF%8D%E0%AE%9F%E0%AF%81%E0%AE%95%E0%AE%B3%E0%AF%8D",
    },
    {
        "source_id": "TVA_BOK_0004641",
        "record_type": "literary_work",
        "title": "ஆத்திசூடி : மூலமும் உரையும்",
        "author": "ஔவையார்",
        "publisher": "Natramizhvilasa Press, Chennai",
        "subject": "Tamil literature",
        "keywords": ["நூல்", "அறம்"],
        "source_url": "https://tamildigitallibrary.in/Articles/%E0%AE%A8%E0%AF%82%E0%AE%B2%E0%AF%8D-4641-%E0%AE%86%E0%AE%A4%E0%AF%8D%E0%AE%A4%E0%AE%BF%E0%AE%9A%E0%AF%82%E0%AE%9F%E0%AE%BF-%E0%AE%AE%E0%AF%82%E0%AE%B2%E0%AE%AE%E0%AF%81%E0%AE%AE%E0%AF%8D%20%E0%AE%89%E0%AE%B0%E0%AF%88%E0%AE%AF%E0%AF%81%E0%AE%AE%E0%AF%8D",
    },
    {
        "source_id": "TVA_BOK_0006466",
        "record_type": "literary_work",
        "title": "நாலடியார்",
        "publisher": "International Institute of Tamil Studies, Chennai",
        "publication_year": 1909,
        "subject": "Tamil literature",
        "holding_institution": "International Institute of Tamil Studies",
        "source_url": "https://tamildigitallibrary.in/Articles/%E0%AE%A8%E0%AF%82%E0%AE%B2%E0%AF%8D-6466-%E0%AE%A8%E0%AE%BE%E0%AE%B2%E0%AE%9F%E0%AE%BF%E0%AE%AF%E0%AE%BE%E0%AE%B0%E0%AF%8D",
    },
    {
        "source_id": "TVA_BOK_0041806",
        "record_type": "literary_work",
        "title": "திருமந்திரம் சொற்பொருளுரை",
        "author": "திருமூலர்",
        "publisher": "Thiruvarul Nilaya Veliyittagam, Kugaioor",
        "publication_year": 2012,
        "subject": "Tamil literature and religion",
        "keywords": ["சைவ சமயம்"],
        "holding_institution": "Department of Tamil Development",
        "source_url": "https://tamildigitallibrary.in/Articles/%E0%AE%A8%E0%AF%82%E0%AE%B2%E0%AF%8D-41806-%E0%AE%A4%E0%AE%BF%E0%AE%B0%E0%AF%81%E0%AE%AE%E0%AE%A8%E0%AF%8D%E0%AE%A4%E0%AE%BF%E0%AE%B0%E0%AE%AE%E0%AF%8D%20%E0%AE%9A%E0%AF%8A%E0%AE%B1%E0%AF%8D%E0%AE%AA%E0%AF%8A%E0%AE%B0%E0%AF%81%E0%AE%B3%E0%AF%81%E0%AE%B0%E0%AF%88",
    },
    {
        "source_id": "TVA_BOK_0003823",
        "record_type": "literary_work",
        "title": "புறநானூறு : மூலமும் உரையும்",
        "author": "புலியூர்க் கேசிகன்",
        "publisher": "Saradha Pathippagam, Chennai",
        "publication_year": 2010,
        "subject": "Sangam literature",
        "keywords": ["எட்டுத்தொகை", "புறம்"],
        "source_url": "https://tamildigitallibrary.in/book-details/TXpneU13PT1BdG9aYXRvejB0bzk-/TlE9PUF0b1phdG96MHRvOQ--/TWpBPUF0b1phdG96MHRvOQ--",
    },
    {
        "source_id": "TVA_BOK_0017042",
        "record_type": "literary_work",
        "title": "நாலடியார் : அரசியல் முதல் ஐந்து அதிகாரங்கள் 134–180, மூலமும் உரையும்",
        "author": "சுப்பிரமணிய சர்மா, நெ. ரா.",
        "publication_year": 1922,
        "subject": "Tamil literature",
        "keywords": ["கல்வி", "குடிப்பிறப்பு", "மேன்மக்கள்"],
        "holding_institution": "Tamil Nadu Archives Library",
        "source_url": "https://tamildigitallibrary.in/Articles/%E0%AE%A8%E0%AF%82%E0%AE%B2%E0%AF%8D-17042-%E0%AE%A8%E0%AE%BE%E0%AE%B2%E0%AE%9F%E0%AE%BF%E0%AE%AF%E0%AE%BE%E0%AE%B0%E0%AF%8D-%E0%AE%85%E0%AE%B0%E0%AE%9A%E0%AE%BF%E0%AE%AF%E0%AE%B2%E0%AF%8D%20%E0%AE%AE%E0%AF%81%E0%AE%A4%E0%AE%B2%E0%AF%8D%20%E0%AE%90%E0%AE%A8%E0%AF%8D%E0%AE%A4%E0%AF%81%20%E0%AE%85%E0%AE%A4%E0%AE%BF%E0%AE%95%E0%AE%BE%E0%AE%B0%E0%AE%99%E0%AF%8D%E0%AE%95%E0%AE%B3%E0%AF%8D%20134%20-%20180",
    },
    {
        "source_id": "TVA_BOK_0008424",
        "record_type": "literary_work",
        "title": "நாலடியார் உரை வளம் : 201–400 பாடல்கள்",
        "publisher": "Saraswathi Mahal Library, Thanjavur",
        "publication_year": 1990,
        "series": "Saraswathi Mahal Library 59 b",
        "subject": "Tamil literature",
        "keywords": ["நாலடியார்", "சமண முனிவர்கள்"],
        "holding_institution": "Saraswathi Mahal Library",
        "source_url": "https://tamildigitallibrary.in/Articles/%E0%AE%A8%E0%AF%82%E0%AE%B2%E0%AF%8D-8424-%E0%AE%A8%E0%AE%BE%E0%AE%B2%E0%AE%9F%E0%AE%BF%E0%AE%AF%E0%AE%BE%E0%AE%B0%E0%AF%8D%20%E0%AE%89%E0%AE%B0%E0%AF%88%20%E0%AE%B5%E0%AE%B3%E0%AE%AE%E0%AF%8D-%20%28%E0%AE%AE%E0%AF%82%E0%AE%B2%E0%AE%AE%E0%AF%81%E0%AE%AE%E0%AF%8D%20%E0%AE%AE%E0%AF%82%E0%AE%A9%E0%AF%8D%E0%AE%B1%E0%AF%81%20%E0%AE%AA%E0%AE%B4%E0%AF%88%E0%AE%AF%20%E0%AE%89%E0%AE%B0%E0%AF%88%E0%AE%95%E0%AE%B3%E0%AF%81%E0%AE%AE%E0%AF%8D%20%E0%AE%85%E0%AE%9F%E0%AE%99%E0%AF%8D%E0%AE%95%E0%AE%BF%E0%AE%AF%E0%AE%A4%E0%AF%81%29",
    },
]


def seed():
    init_db()
    # Remove a source ID typo from the first import; keep the corrected ID below.
    db.tdl_catalog.delete_one({"source_id": "TVA_BOK_00017042"})
    imported_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for record in RECORDS:
        db.tdl_catalog.update_one(
            {"source_id": record["source_id"]},
            {
                "$set": {
                    **record,
                    "source_name": SOURCE,
                    "record_type": record.get("record_type", "catalog_publication"),
                    "imported_at": imported_at,
                }
            },
            upsert=True,
        )
    print(f"Imported {len(RECORDS)} sourced catalog records into {db.name}.tdl_catalog")


if __name__ == "__main__":
    seed()
