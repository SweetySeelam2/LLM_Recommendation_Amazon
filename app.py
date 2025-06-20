# app.py — Final LLM-Based Product Recommendation App (Retrieve → Rank → Explain + SHAP)

import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ------------------------------------------
# ✅ Load Assets from Local Repo (10k only)
# ------------------------------------------

review_embeddings = np.load("review_embeddings_10k.npy")
X_test = np.load("X_test_embeddings_10k.npy")
df = pd.read_csv("amazon_reviews_with_embeddings_10k.csv")
model = XGBRegressor()
model.load_model("model_xgb_regressor.json")
y_test = pd.read_csv("y_test_10k.csv").iloc[:, 0]

# Embedder & Phi-2 LLM (CPU only, required for Streamlit Cloud) 

# --- Load Embedder (Auto handles device)
@st.cache_resource(show_spinner="🔄 Loading Embedder...")
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")  # ✅ No `.to()` used

embedder = load_embedder()

# --- Load Phi-2 Model (SAFE for Streamlit Cloud)
phi2_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2", torch_dtype=torch.float32  # ✅ Avoids meta tensor issue
)
phi2_model.eval()  # ✅ Safe call

# --- Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")

# ------------------------------------------
# 🧠 Session State Utility (ALWAYS fallback to default df)
# ------------------------------------------
def get_current_df():
    return st.session_state["df"] if "df" in st.session_state else df

# ------------------------------------------
# 🔍 Utility Functions (always use current_df)
# ------------------------------------------

def retrieve_top_reviews(query_text, top_n=5):
    current_df = get_current_df()
    q_vec = embedder.encode([query_text])
    sims = cosine_similarity(q_vec, review_embeddings)[0]
    idxs = sims.argsort()[::-1]
    filtered_idxs = [i for i in idxs if current_df.iloc[i]['verified_purchase'] == 1][:top_n]
    top_reviews = current_df.iloc[filtered_idxs]['reviewText'].tolist()
    product_asins = current_df.iloc[filtered_idxs]['asin'].tolist()
    product_name = f"Product ASIN: {product_asins[0]}" if product_asins else "a relevant product"
    return top_reviews, product_name

def generate_phi2_recommendation(query_text, top_reviews, product_name):
    context = "\n\n".join(f"Review {i+1}: {r}" for i, r in enumerate(top_reviews))
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
    inputs = tokenizer(prompt, return_tensors="pt").to('cpu')
    outputs = phi2_model.generate(
        **inputs,
        max_new_tokens=150,
        num_beams=5,
        early_stopping=False,
        length_penalty=0.8,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Recommendation:" in decoded:
        rec_block = decoded.split("Recommendation:")[-1]
    else:
        rec_block = decoded[len(prompt):]
    for line in rec_block.splitlines():
        clean = line.strip()
        if clean and not clean.startswith("#"):
            return clean
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
    st.info("🚀 Note: For speed, this live demo runs on a 10,000-review sample. All model training and business insights use 1,000,000 reviews. See full details on [GitHub](https://github.com/sweetyseelam/llm_recommendation_amazon).") 

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
        st.dataframe(user_df.head(20))
    elif st.button("Use Sample Dataset"):
        st.session_state["df"] = df.copy()
        st.success("✅ Sample dataset loaded!")
        st.dataframe(df.head(20))

    # Always show a preview if available (sample or uploaded)
    current_df = get_current_df()
    if current_df is not None:
        st.dataframe(current_df.head(20))
    st.info("🚀 Note: For speed, this live demo runs on a 10,000-review sample. All model training and business insights use 1,000,000 reviews. See full details on [GitHub](https://github.com/sweetyseelam/llm_recommendation_amazon).")

elif page == "📈 Explain Model (SHAP)":
    st.title("📈 Model Explainability with SHAP")
    st.markdown("""
    This view explains **why** the model predicts certain ratings. The SHAP summary plot shows which features most influence predictions.
    """)
    if st.button("Generate SHAP Summary Plot"):
        try:
            explain_model_with_shap()
        except Exception as e:
            st.error(f"Failed to generate SHAP summary: {e}")
    st.info("🚀 Note: For speed, this live demo runs on a 10,000-review sample. All model training and business insights use 1,000,000 reviews. See full details on [GitHub](https://github.com/sweetyseelam/llm_recommendation_amazon).")
    
elif page == "🤖 LLM Recommendation":
    st.title("🤖 AI-Powered Product Recommendation")
    st.markdown("""
    Enter a product query and get a personalized recommendation based on customer reviews — powered by the **Phi-2 language model**.
    """)

    query = st.text_input("Type your product query here:", "Looking for a long-lasting Bluetooth headset with noise cancellation")
    if st.button("Submit Query"):
        try:
            reviews, product_name = retrieve_top_reviews(query)
            recommendation = generate_phi2_recommendation(query, reviews, product_name)

            st.markdown(f"### 🛍️ Recommended Product: **{product_name}**")
            st.markdown(f"### ✅ Why You'll Love It: {recommendation}")

            st.markdown("#### 🔍 Top 3 Reviews Considered")
            for i, r in enumerate(reviews[:3]):
                st.markdown(f"- **Review {i+1}:** {r}")

            st.info("💡 Note: This recommendation is AI-generated based on verified reviews. It is not an official endorsement. "
                    "🚀 Note: For speed, this live demo runs on a 10,000-review sample. All model training and business insights use 1,000,000 reviews. See full details on [GitHub](https://github.com/sweetyseelam/llm_recommendation_amazon).")
        except Exception as e:
            st.error(f"Failed to generate recommendation: {e}")

# ------------------------------------------
# 📜 MIT License Footer
# ------------------------------------------

st.markdown("""
---
#### © 2025 Sweety Seelam | MIT License
This app and its models are provided under the permissive MIT license. You're free to use, modify, and distribute — with proper credit.
""")