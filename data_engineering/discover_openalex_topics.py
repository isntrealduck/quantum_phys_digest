import getpass
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


# Read your existing OpenAlex key without displaying it.
api_key = getpass.getpass(
    "Paste your OpenAlex API key and press Enter: "
).strip()

if not api_key:
    raise SystemExit("No API key was entered.")


# Search OpenAlex's topic catalogue.
parameters = {
    "search": "quantum",
    "per_page": 25,
}

url = "https://api.openalex.org/topics?" + urlencode(parameters)

request = Request(
    url,
    headers={
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "quantum_phys_digest/0.2",
    },
)


# Download the response.
try:
    with urlopen(request, timeout=60) as response:
        raw_response = response.read()
except HTTPError as error:
    raise SystemExit(
        f"OpenAlex returned HTTP {error.code}. "
        "401/403: check your key. "
        "429: request limit or daily budget reached."
    ) from None
except URLError:
    raise SystemExit(
        "Could not connect to OpenAlex. Check your internet connection."
    ) from None


# Preserve the response in its own timestamped folder.
project_folder = Path(__file__).resolve().parent.parent
retrieved_at = datetime.now(timezone.utc)
run_id = retrieved_at.strftime("%Y%m%dT%H%M%S%fZ")

output_folder = (
    project_folder / "data" / "raw" / "openalex_topics" / run_id
)
output_folder.mkdir(parents=True, exist_ok=False)

(output_folder / "response.json").write_bytes(raw_response)

manifest = {
    "retrieved_at_utc": retrieved_at.isoformat(),
    "endpoint": "https://api.openalex.org/topics",
    "parameters": parameters,
}

(output_folder / "manifest.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)


# Display the topics so we can choose the collection scope.
data = json.loads(raw_response)
topics = data["results"]

print(f"\nTopics matching the search: {data['meta']['count']}")
print(f"Topics displayed: {len(topics)}")

for topic in topics:
    topic_id = topic["id"].rsplit("/", 1)[-1]
    subfield = (topic.get("subfield") or {}).get(
        "display_name", "Unknown"
    )

    print(f"\n{topic_id}: {topic['display_name']}")
    print(f"  Subfield: {subfield}")
    print(f"  Description: {topic.get('description') or 'Unavailable'}")

print(f"\nSaved response: {output_folder / 'response.json'}")