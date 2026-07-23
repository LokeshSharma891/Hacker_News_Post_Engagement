import pandas as pd
import numpy as np

df = pd.read_csv("hn_raw.csv")
df = df.drop_duplicates(subset="id")
df = df[df["author"].notna()]

df["created_dt"] = pd.to_datetime(df["created_utc"], unit="s")
df["hour_of_day"] = df["created_dt"].dt.hour
df["day_of_week"] = df["created_dt"].dt.dayofweek
df["title_len"] = df["title"].str.len()
df["title_word_count"] = df["title"].str.split().str.len()
df["has_question_mark"] = df["title"].str.contains(r"\?").astype(int)
df["has_number"] = df["title"].str.contains(r"\d").astype(int)
df["has_url"] = df["url"].notna().astype(int)
df["log_score"] = np.log1p(df["score"].clip(lower=0))

df.to_csv("hn_features.csv", index=False)
print(df[["title", "score", "hour_of_day", "title_word_count"]].head())
print(f"Rows after cleaning: {len(df)}")