import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

# Describe which papers we want.
params = urlencode({
    "search_query": "cat:quant-ph",
    "start": 0,
    "max_results": 10,
    "sortBy": "submittedDate",
    "sortOrder": "descending",
})

# Construct the request.
request = Request(
    f"https://export.arxiv.org/api/query?{params}",
    headers={"User-Agent": "quantum_phys_digest/0.1"},
)

print("Requesting papers from arXiv...")

# Send the request and read the XML response.
with urlopen(request, timeout=60) as response:
    raw_xml = response.read()

# Save the original response.
output = Path("data/raw")
output.mkdir(parents=True, exist_ok=True)
(output / "sample.xml").write_bytes(raw_xml)

# Parse the XML.
root = ET.fromstring(raw_xml)
ns = {"atom": "http://www.w3.org/2005/Atom"}

papers = []

# Extract useful information from each paper.
for entry in root.findall("atom:entry", ns):
    paper_url = entry.findtext("atom:id", "", ns)

    if "/api/errors" in paper_url:
        raise RuntimeError(entry.findtext("atom:summary", "", ns))

    papers.append({
        "url": paper_url,
        "title": " ".join(
            entry.findtext("atom:title", "", ns).split()
        ),
        "abstract": " ".join(
            entry.findtext("atom:summary", "", ns).split()
        ),
        "submitted_at": entry.findtext("atom:published", "", ns),
        "authors": [
            author.findtext("atom:name", "", ns)
            for author in entry.findall("atom:author", ns)
        ],
    })

if not papers:
    raise RuntimeError("The API returned no papers.")

# Save the extracted information as JSON.
(output / "sample.json").write_text(
    json.dumps(papers, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(f"Saved {len(papers)} papers to {output.resolve()}")
print(f"First paper: {papers[0]['title']}")