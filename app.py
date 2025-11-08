# app.py
import os
import re
import json
import math
import time
import random
import numpy as np
import pandas as pd
import torch
import shap
import matplotlib.pyplot as plt

import streamlit as st
from xgboost import XGBRegressor
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download

# ------------------------------------------------------------
# 🧭 Basic setup
# ------------------------------------------------------------
st.set_page_config(page_title="LLM Product Recommender", layout="wide")
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to:", ["📘 Overview", "📥 Test or Upload Data", "📈 Explain Model (SHAP)", "🤖 LLM Recommendations"])

# Make results reproducible
random.seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.manual_seed(42)

REPO_ID = "sweetyseelam/llm-recommendation-assets"

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def show_footer():
    st.markdown("""
    ---
    #### 🔐 Proprietary & All Rights Reserved                               
    © 2025 Sweety Seelam. All rights reserved. Unauthorized commercial use, redistribution, or duplication of any part of this project is strictly prohibited.
    """)

def clean_html(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<.*?>", " ", text)        # HTML tags
    text = re.sub(r"&nbsp;|&amp;|&quot;", " ", text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize(s: str) -> str:
    return clean_html(s).lower()

def word_count(s: str) -> int:
    return len(clean_html(s).split())

def looks_like_review_sentence(s: str) -> bool:
    # Heuristic: review-y titles are long, often sentence-like, and contain verbs/punctuation
    s_ = clean_html(s)
    return (word_count(s_) > 14) or bool(re.search(r"[.!?]", s_))

def pick_product_name(row: pd.Series, asin: str) -> str:
    # Prefer product_title ∈ [2..15] words, otherwise fallback to shorter title
    for col in ["product_title", "title"]:
        if col in row and isinstance(row[col], str) and row[col].strip():
            cand = clean_html(row[col].strip())
            wc = word_count(cand)
            if 2 <= wc <= 15 and not looks_like_review_sentence(cand):
                return cand
    # Last resort
    for col in ["product_title", "title"]:
        if col in row and isinstance(row[col], str) and row[col].strip():
            return clean_html(row[col].strip())
    return f"Unnamed Product (ASIN: {asin})"

# Build a light “intent lexicon” from the query to steer retrieval
BASE_SYNONYMS = {
    "headset": {"headset", "headsets", "headphone", "headphones", "earbud", "earbuds", "bluetooth"},
    "bluetooth": {"bluetooth", "wireless"},
    "noise": {"noise", "anc", "cancellation", "canceling", "cancelling", "noise-cancelling", "noise-cancellation"},
    "battery": {"battery", "long-lasting", "playtime", "hours", "long", "life"},
    "power bank": {"powerbank", "power-bank", "power", "bank", "portable", "charger", "charging"},
}

STOPWORDS = {
    "a","an","the","and","or","with","for","my","your","their","his","her",
    "of","to","in","on","at","by","from","as","is","are","be","this","that",
    "it","its","you","i","we","they","our","me"
}

def extract_query_terms(q: str) -> set:
    qn = normalize(q)
    tokens = re.findall(r"[a-z0-9\-]+", qn)
    terms = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    termset = set(terms)
    # Expand with synonyms if we detect anchors
    expanded = set(termset)
    for anchor, syns in BASE_SYNONYMS.items():
        if anchor in termset or any(s in termset for s in syns):
            expanded |= syns
    return expanded

def keyword_score(text: str, query_terms: set) -> float:
    """Simple normalized keyword score: proportion of query terms present (ANY-match later)."""
    if not text or not query_terms:
        return 0.0
    words = set(re.findall(r"[a-z0-9\-]+", normalize(text)))
    if not words:
        return 0.0
    hits = len(words & query_terms)
    return hits / max(3, len(query_terms))  # smooth denominator a bit

# ------------------------------------------------------------
# Load assets (robustly)
# ------------------------------------------------------------
@st.cache_resource
def load_static_assets():
    # Embeddings & matrices
    review_embeddings = pd.read_csv(
        hf_hub_download(REPO_ID, filename="review_embeddings_1k_HG.csv", repo_type="dataset")
    ).values

    X_test = pd.read_csv(
        hf_hub_download(REPO_ID, filename="X_test_embeddings_1k_HG.csv", repo_type="dataset")
    ).values

    # Metadata (reviews)
    df = pd.read_csv(
        hf_hub_download(REPO_ID, filename="amazon_reviews_with_embeddings_1k_HG.csv", repo_type="dataset")
    )

    # y_test (optional for SHAP demo)
    y_test = pd.read_csv(
        hf_hub_download(REPO_ID, filename="y_test_1k_HG.csv", repo_type="dataset")
    ).iloc[:, 0].values

    # XGB model
    model = XGBRegressor()
    model.load_model(
        hf_hub_download(REPO_ID, filename="model_xgb_regressor.json", repo_type="dataset")
    )
    return review_embeddings, X_test, df, y_test, model

@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    # Keep Phi-2 on CPU for Spaces free tier stability
    phi2_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2").to("cpu").eval()
    return embedder, tokenizer, phi2_model

review_embeddings, X_test, df, y_test, model = load_static_assets()
embedder, tokenizer, phi2_model = load_models()

# Recompute embeddings if the saved CSV doesn't match current embedder dim (safety)
@st.cache_resource
def ensure_embedding_dim(review_embeddings, df, embedder):
    want_dim = embedder.get_sentence_embedding_dimension()
    have_dim = review_embeddings.shape[1]
    if have_dim == want_dim:
        return review_embeddings
    # Re-encode (only 1k -> fast)
    texts = [clean_html(t) for t in df.get("reviewText", "").astype(str).tolist()]
    enc = embedder.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    return enc

review_embeddings = ensure_embedding_dim(review_embeddings, df, embedder)

# ------------------------------------------------------------
# Session-state utilities
# ------------------------------------------------------------
def get_current_df() -> pd.DataFrame:
    return st.session_state["df"] if "df" in st.session_state else df

# ------------------------------------------------------------
# Retrieval & Recommendation
# ------------------------------------------------------------
def retrieve_top_reviews(query_text: str, top_n: int = 3):
    """Semantic-first, keyword-gated retrieval with hybrid scoring and hard guards."""
    current_df = get_current_df().copy()

    # Verified-only slice
    if "verified_purchase" in current_df.columns:
        current_df = current_df[current_df["verified_purchase"] == 1].copy()

    # Build text fields for matching
    pt = current_df["product_title"] if "product_title" in current_df.columns else pd.Series("", index=current_df.index)
    tt = current_df["title"] if "title" in current_df.columns else pd.Series("", index=current_df.index)
    rt = current_df["reviewText"] if "reviewText" in current_df.columns else pd.Series("", index=current_df.index)

    current_df["full_title"] = (pt.fillna("") + " " + tt.fillna("")).apply(normalize)
    current_df["review_clean"] = rt.fillna("").astype(str).apply(clean_html)
    current_df["asin"] = current_df["asin"].astype(str)

    # Semantic similarity
    q_vec = embedder.encode([query_text], normalize_embeddings=True)
    sims = cosine_similarity(q_vec, review_embeddings)[0]
    current_df["sim"] = sims

    # Lightweight keyword/category gating (ANY match)
    q_terms = extract_query_terms(query_text)
    current_df["kw_score"] = current_df["full_title"].apply(lambda s: keyword_score(s, q_terms))

    # Primary candidate pool: keep rows with either good similarity or any keyword hit
    # Thresholds chosen to be permissive while avoiding junk
    sim_thr = max(0.20, current_df["sim"].quantile(0.60))  # dynamic-ish
    pool = current_df[(current_df["sim"] >= sim_thr) | (current_df["kw_score"] > 0)].copy()

    # If empty, fall back to top-N by sim
    if pool.empty:
        pool = current_df.nlargest(800, "sim").copy()

    # Combine by ASIN using a hybrid score
    pool["hybrid"] = 0.7 * pool["sim"] + 0.3 * pool["kw_score"]
    agg = (
        pool.groupby("asin")
            .agg(mean_sim=("sim", "mean"), max_kw=("kw_score", "max"), mean_h=("hybrid", "mean"), n=("hybrid", "count"))
            .sort_values(["mean_h", "mean_sim", "n"], ascending=False)
    )

    asins_ranked = agg.index.tolist()
    if not asins_ranked:
        # total fallback: strongest rows globally
        rows = current_df.nlargest(3, "sim")
        top_reviews = [clean_html(x) for x in rows["review_clean"].tolist()]
        while len(top_reviews) < top_n:
            top_reviews.append("Review not available.")
        first_row = rows.iloc[0] if len(rows) else pd.Series({})
        product_name = pick_product_name(first_row, asin="UNKNOWN")
        return top_reviews[:top_n], product_name

    # Walk down ASINs, collecting up to top_n reviews. Prefer the best ASIN; borrow from next if needed.
    collected = []
    chosen_asin = None
    for asin in asins_ranked:
        asin_rows = pool[pool["asin"] == asin].sort_values(["hybrid", "sim"], ascending=False)
        for _, r in asin_rows.iterrows():
            rv = clean_html(r.get("review_clean", ""))
            if rv:
                collected.append((asin, r, rv))
                if chosen_asin is None:
                    chosen_asin = asin
            if len([c for c in collected if c[0] == chosen_asin]) >= top_n:
                break
        if chosen_asin and len([c for c in collected if c[0] == chosen_asin]) >= top_n:
            break

    # If the top ASIN still has < top_n, borrow from next best ASINs
    if not chosen_asin:
        chosen_asin = asins_ranked[0]
    top_reviews = [c[2] for c in collected if c[0] == chosen_asin][:top_n]
    if len(top_reviews) < top_n:
        for asin in asins_ranked:
            if asin == chosen_asin:
                continue
            asin_rows = pool[pool["asin"] == asin].sort_values(["hybrid", "sim"], ascending=False)
            for _, r in asin_rows.iterrows():
                rv = clean_html(r.get("review_clean", ""))
                if rv:
                    top_reviews.append(rv)
                    if len(top_reviews) >= top_n:
                        break
            if len(top_reviews) >= top_n:
                break

    while len(top_reviews) < top_n:
        top_reviews.append("Review not available.")

    # Product name from the best row of chosen_asin
    chosen_rows = pool[pool["asin"] == chosen_asin].sort_values(["hybrid", "sim"], ascending=False)
    first_row = chosen_rows.iloc[0]
    product_name = pick_product_name(first_row, chosen_asin)

    return top_reviews[:top_n], product_name

def generate_phi2_recommendation(query_text, top_reviews, product_name):
    # Clean/truncate reviews
    context = "\n\n".join(
        f"Review {i+1}: {clean_html(r)[:300]}" for i, r in enumerate(top_reviews[:3])
    )

    prompt = (
        f"Customer Query: \"{query_text}\"\n\n"
        f"Verified customer reviews for {product_name}:\n\n{context}\n\n"
        f"Write one concise, honest product recommendation for {product_name} that addresses:\n"
        f"- Noise cancellation quality (if applicable)\n"
        f"- Comfort for extended use (battery life implications)\n"
        f"- Overall user satisfaction\n\n"
        f"Start exactly with: 'Based on the customer query and verified reviews, I recommend '.\n"
        f"Avoid generic product names; use the product name given. No hashtags.\n\n"
        "Recommendation:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    outputs = phi2_model.generate(
        **inputs,
        max_new_tokens=140,
        num_beams=5,
        early_stopping=True,
        length_penalty=0.8,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    rec_block = decoded.split("Recommendation:")[-1] if "Recommendation:" in decoded else decoded[len(prompt):]

    # Extract first clean line
    for line in rec_block.splitlines():
        clean = line.strip()
        if clean and not clean.startswith("#"):
            return clean
    return "Based on the customer query and verified reviews, I recommend considering this product. (⚠️ The generator could not craft a detailed reason.)"

# ------------------------------------------------------------
# SHAP explainability (robust to array sizes)
# ------------------------------------------------------------
def explain_model_with_shap():
    # Limit rows for speed
    sample_n = min(200, X_test.shape[0])
    X_sample = X_test[:sample_n]
    try:
        explainer = shap.Explainer(model)
    except Exception:
        # Fallback: TreeExplainer for XGB
        explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)

    feature_names = [f"f{i}" for i in range(X_test.shape[1])]

    # Summary plot
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.gcf().set_size_inches(10, 4.8)
    st.pyplot(plt.gcf())
    plt.clf()

    # Top features by mean |SHAP|
    vals = getattr(shap_values, "values", None)
    if vals is None:
        st.info("SHAP values not directly accessible for this explainer; showing summary only.")
        return

    top_idx = np.argsort(np.abs(vals).mean(0))[::-1][:5]
    top_feature_names = [feature_names[i] for i in top_idx]

    st.markdown("#### 📌 General SHAP Explanation")
    st.markdown("""
    SHAP (SHapley Additive exPlanations) attributes how much each feature pushes a prediction up or down.
    - **Red** points indicate higher feature values; **blue** indicate lower values.
    - Points to the right increase predicted rating; to the left decrease it.
    """)

    st.markdown("#### 📊 Results Interpretation")
    st.markdown(f"""
    From the summary for the sampled rows (n={sample_n}), the **most influential features** include:
    **{', '.join(top_feature_names)}**.
    Use these to understand what most drives predicted satisfaction in the dataset.
    """)

# ------------------------------------------------------------
# PAGES
# ------------------------------------------------------------
if page == "📘 Overview":
    st.title("📘 Project Overview")
    st.markdown("""
    This professional-grade app helps users find the **best product** for their query through a 3-stage pipeline:

    - ✅ **Retrieve** similar reviews via sentence embeddings  
    - ✅ **Rank** candidates with a hybrid semantic–keyword score + XGBoost signals  
    - ✅ **Explain** predictions with SHAP and craft a natural recommendation with **Phi-2**

    **Built With:** SentenceTransformers • XGBoost • SHAP • Hugging Face Phi-2 • Streamlit
    """)
    st.info("🚀 Note: The live demo uses a 1,000-review sample for speed. Full training used a larger corpus. See: [GitHub](https://github.com/SweetySeelam2/LLM_Recommendation_Amazon) or the Space README.")
    show_footer()

elif page == "📥 Test or Upload Data":
    st.title("📥 Test the Model with Our Sample or Upload Your Own Data")
    st.markdown("""
    **How to Use**
    1) Upload a CSV with columns: `reviewText`, `verified_purchase`, `helpful_vote`, `asin`  
       **or** click **Use Sample Dataset**.  
    2) Click the appropriate **Submit** button to generate predictions and a downloadable file.
    """)

    uploaded = st.file_uploader("📤 Upload your CSV", type="csv")

    if uploaded:
        try:
            user_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"File could not be read: {e}")
            user_df = None

        required = {"reviewText", "verified_purchase", "helpful_vote", "asin"}
        if user_df is not None:
            if not required.issubset(set(user_df.columns)):
                st.error("❌ Missing columns. Required: `reviewText`, `verified_purchase`, `helpful_vote`, `asin`.")
            else:
                st.session_state["df"] = user_df
                st.success("✅ Uploaded successfully!")
                st.dataframe(user_df.head(20))

                if st.button("📩 Submit Uploaded Data"):
                    try:
                        preds = np.round(model.predict(X_test)[:len(user_df)]).astype(int)
                        df_preds = user_df.copy()
                        df_preds["Predicted Rating"] = preds
                        st.session_state["df_preds"] = df_preds
                        st.success("✅ Predictions generated!")
                        st.dataframe(df_preds.head(20))
                        st.download_button("📥 Download Prediction Results", df_preds.to_csv(index=False), file_name="predictions_uploaded.csv")

                        st.markdown("### 📊 Results Interpretation")
                        st.markdown("""
                        - Predicted review ratings (1–5) are inferred from learned patterns (e.g., verified purchase, helpful votes).
                        - Use these to gauge likely satisfaction trends in your data.
                        """)
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")

    if st.button("📦 Use Sample Dataset"):
        st.session_state["df"] = df.copy()
        st.session_state["use_sample"] = True
        st.success("✅ Sample dataset loaded!")
        st.dataframe(df.head(20))

    if st.session_state.get("use_sample"):
        if st.button("📩 Submit Sample Dataset"):
            try:
                preds = np.round(model.predict(X_test)).astype(int)
                df_preds = df.copy()
                df_preds["Predicted Rating"] = preds
                st.session_state["df_preds"] = df_preds
                st.success("✅ Predictions generated for sample!")
                st.dataframe(df_preds.head(20))
                st.download_button("📥 Download Predictions Results", df_preds.to_csv(index=False), file_name="predictions_sample.csv")

                st.markdown("""
                ---
                ### 📊 Results Interpretation:
                - These are **model-predicted review ratings** (scale 1–5).
                - Use this benchmark before uploading your own data.
                """)
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    st.markdown("---")
    st.markdown("**Next:** go to **📈 Explain Model (SHAP)** to see feature influence.")
    show_footer()

elif page == "📈 Explain Model (SHAP)":
    st.title("📈 Model Explainability with SHAP")
    st.markdown("""
    Click the button to compute a **SHAP summary plot** for a sample of rows.
    """)
    if "df_preds" not in st.session_state:
        st.warning("⚠️ Submit data on **📥 Test or Upload Data** first.")
    else:
        if st.button("📊 Generate SHAP Summary Plot"):
            try:
                explain_model_with_shap()
            except Exception as e:
                st.error(f"Failed to generate SHAP: {e}")
    show_footer()

elif page == "🤖 LLM Recommendations":
    st.title("🤖 AI-Powered Product Recommendation")
    st.markdown("""
    **How to Use**
    - Enter a product query (e.g., *“long-lasting Bluetooth headset with noise cancellation”*).  
    - The app will retrieve similar **verified** reviews, pick the best product, and write a concise suggestion with **Phi-2**.
    """)

    default_q = "Looking for a long-lasting Bluetooth headset with noise cancellation"
    query = st.text_input("🔎 Enter your product query here:", default_q)

    if st.button("🤖 Submit Query"):
        try:
            reviews, product_name = retrieve_top_reviews(query, top_n=3)
            recommendation = generate_phi2_recommendation(query, reviews, product_name)

            st.markdown(f"### 🛍️ Recommended Product: **{product_name}**")
            st.markdown(f"### ✅ Why You'll Love It\n{recommendation}")

            st.markdown("#### 🔍 Top 3 Reviews Considered")
            for i, r in enumerate(reviews[:3]):
                cleaned_review = clean_html(r)
                st.markdown(f"- **Review {i+1}:** {cleaned_review if cleaned_review else 'Review not available.'}")

            st.markdown("""
            ---
            **Note:** This is a language-model suggestion synthesized from verified user reviews.
            It’s intended to help you explore suitable options — not a paid endorsement.
            """)
        except Exception as e:
            st.error(f"❌ Failed to generate recommendation: {e}")

    show_footer()