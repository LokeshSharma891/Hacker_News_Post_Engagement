# Hacker News Post Engagement Predictor

Predicts how much engagement (score) a Hacker News post is likely to get, based on its title and posting details — and explains *why*, using SHAP feature importance.

🔗 **Live app:** [ADD YOUR STREAMLIT URL HERE, e.g. https://your-app-name.streamlit.app]

---

## The question

Does a post's title, length, posting time, or format actually predict how much engagement it gets on Hacker News — and if so, which of those matters most?

## Data

- **Source:** [Hacker News API](https://github.com/HackerNews/API) and the [HN Algolia Search API](https://hn.algolia.com/api), both free and public, no authentication required.
- **Collection method:** pulled stories in daily time windows over a ~90-day period via the Algolia search endpoint, de-duplicated by post ID.
- **Fields collected:** title, score, number of comments, timestamp, author, URL presence, and whether the post was an "Ask HN" thread.
- **Size:** 81,060 posts after cleaning (17 columns of features and metadata per post)

## Approach

1. **Feature engineering** — extracted posting hour, day of week, title length and word count, presence of a question mark or number in the title, whether the post linked out or was text-only, and whether it was an "Ask HN" post.
2. **Baseline** — a naive "always predict the median score" model, to have an honest floor to beat.
3. **Model** — a LightGBM regressor trained on log-transformed scores (scores are heavily right-skewed, so raw values would distort the loss function).
4. **Explainability** — used SHAP to identify which features actually drove predictions, rather than treating the model as a black box.
5. **Deployment** — wrapped the trained model in a Streamlit app so anyone can type a draft title and posting time and get a live predicted score.

## Results

- Model MAE (LightGBM): 0.74 (on log-transformed score; lower is better)
- Baseline MAE (median predictor): [FILL IN — rerun `train_model.py` and copy the "Baseline MAE" line it prints, so the comparison is complete]
- Top features by SHAP importance: hour of day posted, title length, and day of week were the three strongest predictors of engagement — posting time and structural title properties mattered more than any single word choice in the title.

*(Open `shap_summary.png` in this repo for the full feature-importance plot.)*

## What didn't work / limitations

- Timing and structural features (when and how long a title is) outweighed content-based signals in this model — question marks and numbers in titles had comparatively little individual impact once posting time and length were accounted for.
- 81,060 posts span roughly the last ~90 days at the time of collection, so the model reflects recent HN dynamics rather than long-term historical trends.
- Engagement on Hacker News is also driven by factors this model can't see — who happens to be online, front-page algorithm placement, and pure luck — so this predicts a *tendency*, not a guarantee.

## Project structure

```
project-ds/
├── collect_data.py       # Pulls raw posts from the HN Algolia API
├── build_features.py     # Cleans data and engineers features
├── train_model.py        # Trains the LightGBM model, reports MAE vs baseline
├── explain_model.py       # Generates SHAP feature-importance plot
├── app.py                 # Streamlit app for live predictions
├── requirements.txt       # Dependencies for local + cloud deployment
├── hn_raw.csv              # Raw collected data
├── hn_features.csv         # Cleaned, feature-engineered data
├── model.pkl               # Trained model
└── shap_summary.png        # Feature importance plot
```

## Running it locally

```bash
git clone https://github.com/LokeshSharma891/Hacker_News_Post_Engagement.git
cd Hacker_News_Post_Engagement
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
streamlit run app.py
```

## Tools used

Python, pandas, LightGBM, scikit-learn, SHAP, Streamlit, Hacker News API / Algolia Search API

## Next steps

- Since posting time (hour, day of week) is one of the strongest predictors, a natural extension is a "best time to post" recommendation feature in the app itself.
- Add text embeddings (via `sentence-transformers`) to see whether semantic meaning in the title adds predictive power beyond structural features like length.
- Extend to compare engagement patterns across HN vs. Reddit if Reddit API access becomes available later.
