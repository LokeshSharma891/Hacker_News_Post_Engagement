import streamlit as st
import joblib
import pandas as pd

model = joblib.load("model.pkl")
features = ["hour_of_day", "day_of_week", "title_len", "title_word_count",
            "has_question_mark", "has_number", "has_url", "is_ask_hn"]

st.title("Hacker News Post Engagement Predictor")
title = st.text_input("Draft your post title:")
hour = st.slider("Hour of day you'll post (0-23)", 0, 23, 12)
day = st.selectbox("Day of week", ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
has_url = st.checkbox("Link post (not text-only)?", value=True)

if title:
    row = pd.DataFrame([{
        "hour_of_day": hour,
        "day_of_week": ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"].index(day),
        "title_len": len(title),
        "title_word_count": len(title.split()),
        "has_question_mark": int("?" in title),
        "has_number": int(any(c.isdigit() for c in title)),
        "has_url": int(has_url),
        "is_ask_hn": int(title.strip().lower().startswith("ask hn")),
    }])
    pred_log_score = model.predict(row[features])[0]
    st.metric("Predicted score", f"{round(2.718**pred_log_score - 1)}")