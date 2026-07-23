import requests
import pandas as pd
import time
from datetime import datetime, timedelta

rows = []
seen_ids = set()

# Pull front-page-worthy stories in daily windows over the last ~90 days.
# Each window is one API call returning up to 1000 hits, so this scales fast.
end_date = datetime.utcnow()
for days_back in range(0, 90):
    day = end_date - timedelta(days=days_back)
    start_ts = int(datetime(day.year, day.month, day.day).timestamp())
    end_ts = start_ts + 86400

    resp = requests.get(
        "https://hn.algolia.com/api/v1/search_by_date",
        params={
            "tags": "story",
            "numericFilters": f"created_at_i>={start_ts},created_at_i<{end_ts}",
            "hitsPerPage": 1000,
        },
    ).json()

    for hit in resp.get("hits", []):
        if hit["objectID"] in seen_ids:
            continue
        seen_ids.add(hit["objectID"])
        rows.append({
            "id": hit["objectID"],
            "title": hit.get("title"),
            "score": hit.get("points", 0),
            "num_comments": hit.get("num_comments", 0),
            "created_utc": hit.get("created_at_i"),
            "author": hit.get("author"),
            "url": hit.get("url"),
            "is_ask_hn": (hit.get("title") or "").startswith("Ask HN"),
        })
    print(f"Day -{days_back}: {len(rows)} rows collected so far")
    time.sleep(0.2)  # be polite to the free API

df = pd.DataFrame(rows)
df = df[df["title"].notna()]
df.to_csv("hn_raw.csv", index=False)
print(f"Done. Total rows: {len(df)}")