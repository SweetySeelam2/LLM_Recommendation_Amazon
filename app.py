# app.py — Final LLM-Based Product Recommendation App (Retrieve → Rank → Explain + SHAP)

import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM
import gdown
import os

def download_from_gdrive(file_id, dest):
    if not os.path.exists(dest):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, dest, quiet=False)

# -- Use ONLY file IDs below (not full URLs) --
REVIEW_EMBEDDINGS_ID = "1y1IIWbihrPox6Lv3vHj0u9LdumpoiOhM"
XTEST_EMBEDDINGS_ID = "1eCvwH4eLogPt6H57bfJ8L_AMCcWjGTjO"
REVIEWS_CSV_ID     = "17JO8WAefDoJeGZh9dN19k0V33qSac6e6"
MODEL_PKL_ID       = "1KoWzyLyHV3Dd4bMvPDWcpjgUCMq9Wk94"
Y_TEST_ID = "1UAze0cRUyVXxBKhVcwINtDzK5vPhyMFv"

download_from_gdrive(REVIEW_EMBEDDINGS_ID, "review_embeddings.npy")
download_from_gdrive(XTEST_EMBEDDINGS_ID, "X_test_embeddings.npy")
download_from_gdrive(REVIEWS_CSV_ID, "amazon_reviews_with_embeddings_v1.csv")
download_from_gdrive(MODEL_PKL_ID, "model_xgb_regressor.pkl")
download_from_gdrive(Y_TEST_ID, "y_test.csv")

# Now load as before
review_embeddings = np.load("review_embeddings.npy")
X_test = np.load("X_test_embeddings.npy")
df = pd.read_csv("amazon_reviews_with_embeddings_v1.csv")
model = joblib.load("model_xgb_regressor.pkl")
y_test = pd.read_csv("y_test.csv").iloc[:, 0]   # Load from local file

# Embedder & Phi-2
embedder = SentenceTransformer("all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
phi2_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
phi2_model.to(device).eval()

# ------------------------------------------
# 🔍 Utility Functions
# ------------------------------------------

def retrieve_top_reviews(query_text, top_n=5):
    q_vec = embedder.encode([query_text])
    sims = cosine_similarity(q_vec, review_embeddings)[0]
    idxs = sims.argsort()[::-1]
    filtered_idxs = [i for i in idxs if df.iloc[i]['verified_purchase'] == 1][:top_n]
    top_reviews = df.iloc[filtered_idxs]['reviewText'].tolist()
    product_asins = df.iloc[filtered_idxs]['asin'].tolist()
    product_name = f"Product ASIN: {product_asins[0]}" if product_asins else "a relevant product"
    return top_reviews, product_name


def generate_phi2_recommendation(query_text, top_reviews, product_name):
    # Build the review context
    context = "\n\n".join(f"Review {i+1}: {r}" for i, r in enumerate(top_reviews))

    # Sharpened prompt with explicit bullet points
    prompt = (
        f"Customer Query: \"{query_text}\"\n\n"
        f"Here are some verified customer reviews:\n{context}\n\n"
        f"Write a single, concise product recommendation for {product_name},\n"
        "covering these three aspects:\n"
        "  1. Noise cancellation quality\n"
        "  2. Comfort for extended use (implying battery life)\n"
        "  3. Overall user satisfaction\n\n"
        "Recommendation:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = phi2_model.generate(
        **inputs,
        max_new_tokens=150,
        num_beams=5,
        early_stopping=False,
        length_penalty=0.8,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    # Decode full output
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the part after "Recommendation:"
    if "Recommendation:" in decoded:
        rec_block = decoded.split("Recommendation:")[-1]
    else:
        rec_block = decoded[len(prompt):]

    # Clean and return the first meaningful line
    for line in rec_block.splitlines():
        clean = line.strip()
        if clean and not clean.startswith("#"):
            return clean

    # Fallback
    return "⚠️ Unable to generate a clear recommendation."


def explain_model_with_shap():
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test[:200])
    feature_names = [f"f{i}" for i in range(X_test.shape[1])]
    fig = plt.figure()
    shap.summary_plot(shap_values, X_test[:200], feature_names=feature_names, show=False)
    st.pyplot(fig)

# ------------------------------------------
# 🧠 Streamlit App — Multipage Layout
# ------------------------------------------

st.set_page_config(page_title="LLM Product Recommender", layout="wide")
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to:", ["📘 Overview", "📥 Test or Upload Data", "📈 Explain Model (SHAP)", "🤖 LLM Recommendation"])

if page == "📘 Overview":
    st.title("📘 Project Overview")
    st.markdown("""
    This professional-grade app helps users find the **best product** based on their query, using a powerful pipeline:

    - ✅ **Retrieve** similar product reviews using sentence embeddings
    - ✅ **Rank** them using a trained XGBoost regression model
    - ✅ **Explain** predictions using SHAP and generate an LLM-based natural recommendation using **Phi-2**

    ---
    **Use Cases:**
    - Boosting product search experiences
    - Personalized shopping assistants
    - Enhancing review analytics for businesses

    **Built With:** XGBoost • SHAP • Hugging Face Phi-2 • SentenceTransformers • Streamlit
    """)

elif page == "📥 Test or Upload Data":
    st.title("📥 Test the Model with Our Sample or Upload Your Own Data")
    st.markdown("""
    You can either:
    - Use our **sample Amazon review dataset**, or
    - Upload your own CSV with these **required columns**:
      - `reviewText` (text), `verified_purchase` (0 or 1), `helpful_vote` (int), `asin` (text)
    """)

    uploaded = st.file_uploader("Upload your CSV file here", type="csv")
    if uploaded:
        user_df = pd.read_csv(uploaded)
        st.session_state["df"] = user_df
        st.success("✅ Uploaded and ready!")
    else:
        if st.button("Use Sample Dataset"):
            st.session_state["df"] = df.copy()
            st.success("✅ Sample dataset loaded!")

elif page == "📈 Explain Model (SHAP)":
    st.title("📈 Model Explainability with SHAP")
    st.markdown("""
    This view explains **why** the model predicts certain ratings. The SHAP summary plot shows which features most influence predictions.
    """)
    if st.button("Generate SHAP Summary Plot"):
        explain_model_with_shap()

elif page == "🤖 LLM Recommendation":
    st.title("🤖 AI-Powered Product Recommendation")
    st.markdown("""
    Enter a product query and get a personalized recommendation based on customer reviews — powered by the **Phi-2 language model**.
    """)

    query = st.text_input("Type your product query here:", "Looking for a long-lasting Bluetooth headset with noise cancellation")
    if st.button("Submit Query"):
        reviews, product_name = retrieve_top_reviews(query)
        recommendation = generate_phi2_recommendation(query, reviews, product_name)

        st.markdown(f"### 🛍️ Recommended Product: **{product_name}**")
        st.markdown(f"### ✅ Why You'll Love It: {recommendation}")

        st.markdown("#### 🔍 Top 3 Reviews Considered")
        for i, r in enumerate(reviews[:3]):
            st.markdown(f"- **Review {i+1}:** {r}")

        st.info("💡 Note: This recommendation is AI-generated based on verified reviews. It is not an official endorsement.")

# ------------------------------------------
# 📜 MIT License Footer
# ------------------------------------------

st.markdown("""
---
#### © 2025 Sweety Seelam | MIT License
This app and its models are provided under the permissive MIT license. You're free to use, modify, and distribute — with proper credit.
""")