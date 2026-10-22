"""
utils/exporter.py
Generates a plain-text and HTML report for a manuscript record.
No external PDF library needed — HTML can be printed to PDF from browser.
"""
from datetime import datetime


def build_html_report(m: dict, tags: list, artifacts: list) -> str:
    tag_html = "".join(
        f'<span class="tag">{t["tag"]} ({t["category"]})</span>' for t in tags
    ) or "<em>None detected</em>"

    artifact_rows = "".join(
        f"<tr><td>{a['name']}</td><td>{a['description']}</td><td>{a['symbol']}</td></tr>"
        for a in artifacts
    ) or "<tr><td colspan='3'><em>No artifacts recorded</em></td></tr>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Manuscript Report — {m['title']}</title>
<style>
  body {{ font-family: 'Segoe UI', serif; max-width: 800px; margin: 2rem auto; color: #2c1a00; }}
  h1   {{ color: #6b4c11; border-bottom: 2px solid #d4c9a8; padding-bottom: 0.5rem; }}
  h2   {{ color: #a0722a; margin-top: 1.5rem; }}
  table{{ border-collapse: collapse; width: 100%; margin-top: 0.5rem; }}
  td, th {{ border: 1px solid #d4c9a8; padding: 0.5rem 0.75rem; text-align: left; }}
  th   {{ background: #f5f0e8; }}
  .tamil {{ font-family: 'Noto Sans Tamil', 'Latha', serif; font-size: 1.1rem;
            line-height: 1.9; background: #f5f0e8; padding: 1rem;
            border: 1px solid #d4c9a8; border-radius: 6px; white-space: pre-wrap; }}
  .tag {{ display: inline-block; background: #e8dfc8; color: #6b4c11;
          padding: 0.2rem 0.6rem; border-radius: 20px; font-size: 0.85rem;
          margin: 0.2rem; }}
  .footer {{ margin-top: 2rem; font-size: 0.8rem; color: #7a6a50; border-top: 1px solid #d4c9a8; padding-top: 0.5rem; }}
  @media print {{ body {{ margin: 1rem; }} }}
</style>
</head>
<body>
<h1>🏺 {m['title']}</h1>
<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

<h2>📋 Metadata</h2>
<table>
  <tr><th>Field</th><th>Value</th></tr>
  <tr><td>Location</td><td>{m.get('location','Unknown')}</td></tr>
  <tr><td>Dynasty</td><td>{m.get('dynasty','Unknown')}</td></tr>
  <tr><td>Period</td><td>{m.get('period','Unknown')}</td></tr>
  <tr><td>Category</td><td>{m.get('category','Unknown')}</td></tr>
  <tr><td>Script Style</td><td>{m.get('style','Unknown')}</td></tr>
  <tr><td>Confidence</td><td>{int((m.get('confidence') or 0) * 100)}%</td></tr>
  <tr><td>Digitized On</td><td>{str(m.get('created_at',''))[:16]}</td></tr>
</table>

<h2>📜 Extracted Tamil Text</h2>
<div class="tamil">{m.get('clean_text') or '[No text extracted]'}</div>

<h2>🏷️ Knowledge Tags</h2>
<div>{tag_html}</div>

<h2>🏺 Artifacts</h2>
<table>
  <tr><th>Name</th><th>Description</th><th>Symbol</th></tr>
  {artifact_rows}
</table>

<div class="footer">
  Tamil Palm-Leaf Manuscript Digitizer &amp; Intelligent Analysis System
</div>
</body>
</html>"""


def build_text_report(m: dict, tags: list) -> str:
    tag_str = ", ".join(f"{t['tag']} ({t['category']})" for t in tags) or "None"
    return f"""TAMIL PALM-LEAF MANUSCRIPT REPORT
==================================
Title    : {m['title']}
Location : {m.get('location','Unknown')}
Dynasty  : {m.get('dynasty','Unknown')}
Period   : {m.get('period','Unknown')}
Category : {m.get('category','Unknown')}
Style    : {m.get('style','Unknown')}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

EXTRACTED TAMIL TEXT
--------------------
{m.get('clean_text') or '[No text extracted]'}

KNOWLEDGE TAGS
--------------
{tag_str}

--
Tamil Palm-Leaf Manuscript Digitizer & Intelligent Analysis System
"""
