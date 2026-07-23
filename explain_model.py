import pandas as pd
import joblib
import shap

df = pd.read_csv("hn_features.csv")
features = ["hour_of_day", "day_of_week", "title_len", "title_word_count",
            "has_question_mark", "has_number", "has_url", "is_ask_hn"]
model = joblib.load("model.pkl")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(df[features])

shap.summary_plot(shap_values, df[features], show=False)
import matplotlib.pyplot as plt
plt.savefig("shap_summary.png", bbox_inches="tight")
print("Saved shap_summary.png — open it and see which features matter most.")