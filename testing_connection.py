import requests

top_ids = requests.get("https://hacker-news.firebaseio.com/v0/topstories.json").json()
print(f"Got {len(top_ids)} story IDs")

item = requests.get(f"https://hacker-news.firebaseio.com/v0/item/{top_ids[0]}.json").json()
print(item["title"], "| score:", item.get("score"))