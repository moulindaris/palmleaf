from mcp.server.fastmcp import FastMCP
from database.db import (
    db, get_manuscript, get_locations, init_db,
    search_manuscripts as search_catalogue,
)

mcp = FastMCP("TamilManuscriptPreservationServer")
init_db()

@mcp.tool()
def search_manuscripts(keyword: str = "", dynasty: str = "", subject: str = "") -> list:
    """Search ancient Tamil manuscripts by keyword, dynasty, or subject."""
    return search_catalogue(keyword, subject, dynasty)[:10]

@mcp.tool()
def get_manuscript_details(manuscript_id: str) -> dict:
    """Retrieve full manuscript record including OCR, cleaned text, artifacts."""
    try:
        doc = get_manuscript(int(manuscript_id))
    except ValueError:
        doc = None
    return doc or {"error": f"Manuscript {manuscript_id} not found."}

@mcp.tool()
def get_historical_locations(dynasty: str = "") -> list:
    """Retrieve ancient excavation and modern repository locations."""
    return [loc for loc in get_locations()
            if not dynasty or dynasty.casefold() in str(loc.get("dynasties_associated", "")).casefold()]

@mcp.tool()
def get_mindmap_nodes(manuscript_id: str) -> dict:
    """Generate nodes and edges for knowledge graph visualization."""
    try:
        doc = get_manuscript(int(manuscript_id))
    except ValueError:
        doc = None
    if not doc:
        return {"error": "Not found"}
    nodes = [
        {"id": "root", "label": doc.get("title", "Manuscript"), "type": "root"},
        {"id": "dynasty", "label": f"Dynasty: {doc.get('dynasty')}", "type": "dynasty"},
        {"id": "subject", "label": f"Subject: {doc.get('subject')}", "type": "subject"},
        {"id": "script", "label": f"Script: {doc.get('script')}", "type": "script"},
    ]
    edges = [
        {"source": "root", "target": "dynasty"},
        {"source": "root", "target": "subject"},
        {"source": "root", "target": "script"},
    ]
    return {"nodes": nodes, "edges": edges}

if __name__ == "__main__":
    mcp.run()
