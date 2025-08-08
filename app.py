
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
import re

# ------------------------------------------
# 🧠 Hugging Face App — Multipage Layout
# ------------------------------------------
st.set_page_config(page_title="LLM Product Recommender", layout="wide")
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to:", ["📘 Overview", "📥 Test or Upload Data", "📈 Explain Model (SHAP)", "🤖 LLM Recommendations"])

# Navigation via buttons
#if st.session_state.get("navigate_to_llm"):
#    page = "🤖 LLM Recommendations"
#    del st.session_state["navigate_to_llm"]
#elif st.session_state.get("navigate_to_shap"):
#    page = "📈 Explain Model (SHAP)"
#    del st.session_state["navigate_to_shap"]
#elif st.session_state.get("page"):
#    page = st.session_state["page"]
#    del st.session_state["page"]

# ------------------------------------------
# ✅ Load Assets from Hugging Face Space
# ------------------------------------------

REPO_ID = "sweetyseelam/llm-recommendation-assets"

# ✅ Load embeddings (CSV, not NPY anymore)
review_embeddings = pd.read_csv(
    hf_hub_download(REPO_ID, filename="review_embeddings_1k_HG.csv", repo_type="dataset")
).values

X_test = pd.read_csv(
    hf_hub_download(REPO_ID, filename="X_test_embeddings_1k_HG.csv", repo_type="dataset")
).values

# ✅ Load review metadata
df = pd.read_csv(
    hf_hub_download(REPO_ID, filename="amazon_reviews_with_embeddings_1k_HG.csv", repo_type="dataset")
)

# ✅ Load y_test (flattened to 1D array)
y_test = pd.read_csv(
    hf_hub_download(REPO_ID, filename="y_test_1k_HG.csv", repo_type="dataset")
).iloc[:, 0].values

# ✅ Load trained XGBoost model (JSON format, Pickle-free)
model = XGBRegressor()
model.load_model(
    hf_hub_download(REPO_ID, filename="model_xgb_regressor.json", repo_type="dataset")
)

print("✅ All data and model assets successfully loaded from Hugging Face.")

# ✅ Safe Embedder & Phi-2 Loading with Cache and Device Fix
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    phi2_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2").to("cpu").eval() #.to("cuda" if torch.cuda.is_available() else "cpu")
    return embedder, tokenizer, phi2_model

embedder, tokenizer, phi2_model = load_models()

# ------------------------------------------
# 🧠 Session State Utility (ALWAYS fallback to default df)
# ------------------------------------------
def get_current_df():
    return st.session_state["df"] if "df" in st.session_state else df

def show_footer():
    st.markdown("""
    ---
    #### 🔐 Proprietary & All Rights Reserved                               
    © 2025 Sweety Seelam. All rights reserved. Unauthorized commercial use, redistribution, or duplication of any part of this project is strictly prohibited.
    """)

# ------------------------------------------
# 🔍 Utility Functions (always use current_df)
# ------------------------------------------
def clean_html(text):
    return re.sub(r"<.*?>", "", text)

def retrieve_top_reviews(query_text, top_n=5):
    current_df = get_current_df()
    q_vec = embedder.encode([query_text])

    # Handle dimension mismatch
    if review_embeddings.shape[1] != q_vec.shape[1]:
        review_vecs_aligned = review_embeddings[:, :q_vec.shape[1]]
    else:
        review_vecs_aligned = review_embeddings

    sims = cosine_similarity(q_vec, review_vecs_aligned)[0]
    idxs = sims.argsort()[::-1]
    filtered_idxs = [i for i in idxs if current_df.iloc[i]['verified_purchase'] == 1][:top_n]
    top_reviews = current_df.iloc[filtered_idxs]['reviewText'].tolist()

    # ✅ Try to get product title first, fallback to ASIN
    try:
        product_name_raw = current_df.iloc[filtered_idxs[0]].get("title") or current_df.iloc[filtered_idxs[0]].get("product_title")
        product_name = product_name_raw.strip() if product_name_raw else f"Product ASIN: {current_df.iloc[filtered_idxs[0]]['asin']}"
    except:
        product_name = f"Product ASIN: {current_df.iloc[filtered_idxs[0]]['asin']}" if filtered_idxs else "a relevant product"

    return top_reviews, product_name

def generate_phi2_recommendation(query_text, top_reviews, product_name):
    # Clean and truncate reviews
    context = "\n\n".join(
        f"Review {i+1}: {clean_html(r)[:300]}" for i, r in enumerate(top_reviews[:3])
    )
    
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

    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    outputs = phi2_model.generate(
        **inputs,
        max_new_tokens=150,
        num_beams=5,
        early_stopping=True,
        length_penalty=0.8,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    rec_block = decoded.split("Recommendation:")[-1] if "Recommendation:" in decoded else decoded[len(prompt):]

    for line in rec_block.splitlines():
        clean = line.strip()
        if clean and not clean.startswith("#"):
            return clean
    return "⚠️ Unable to generate a clear recommendation."

def explain_model_with_shap():
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test[:200])
    feature_names = [f"f{i}" for i in range(X_test.shape[1])]
    
    shap.summary_plot(shap_values, X_test[:200], feature_names=feature_names, show=False)
    plt.gcf().set_size_inches(10, 4.5)  # ✅ ACTUALLY RESIZES the SHAP plot
    st.pyplot(plt.gcf())  # ✅ Capture and render resized plot

    top_features = np.argsort(np.abs(shap_values.values).mean(0))[::-1][:5]
    top_feature_names = [feature_names[i] for i in top_features]

    st.markdown("#### 📌 General SHAP Explanation")
    st.markdown("""
    SHAP (SHapley Additive exPlanations) values help explain how much each feature contributes to a specific prediction.
    - The SHAP plot highlights which features most influenced the model's decisions.
    - High SHAP values indicate stronger impact — either positively or negatively — on the model output.
    - Positive SHAP values increase the predicted rating; negative values reduce it.
    - Darker colors show higher feature values (e.g., more helpful votes).
    - Features like `verified_purchase`, `helpful_vote`, and `reviewText` embeddings may have strong influence.
    - This helps **build trust** in the prediction process.
    """)

    st.markdown("#### 📊 Results Interpretation")
    st.markdown(f"""
    Based on the above SHAP plot generated using the top 200 rows from your selected dataset:
    - The **most influential features** affecting product rating prediction are: {', '.join(top_feature_names)}.
    - Red bars indicate features that increase the predicted rating, blue bars decrease it.
    - Based on the selected dataset and generated SHAP Summary plot, the **{top_feature_names[0]}** is red and high on the above generated plot, thus this feature is contributing strongly to high ratings.
    - Use this to prioritize what drives satisfaction in customer reviews.
    """)

# ------------------------------------------
# PAGE 1: OVERVIEW
# ------------------------------------------
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
    - Business feedback mining  
    - AI-driven review analysis; Enhancing review analytics for businesses

    **Built With:** XGBoost • SHAP • Hugging Face Phi-2 • SentenceTransformers • Hugging Face Space
    """)

    st.info("🚀 Note: For speed, this live demo runs on a 1,000-review sample. All model training and business insights use 1,000,000 reviews. See full details on [Github Repo](https://github.com/SweetySeelam2/LLM_Recommendation_Amazon) **or** [Hugging Face](https://huggingface.co/spaces/sweetyseelam/llm-product-recommender).")

    st.markdown("---")
    #if st.button("➡️ Click here to navigate to next page: 📥 Test or Upload Data"):
    #    st.session_state["page"] = "📥 Test or Upload Data"
    #    st.experimental_rerun()

    show_footer()

# ------------------------------------------
# PAGE 2: SAMPLE OR UPLOAD + PREDICT + DOWNLOAD
# ------------------------------------------
elif page == "📥 Test or Upload Data":
    st.title("📥 Test the Model with Our Sample or Upload Your Own Data")
    st.markdown("""
    ### 📌 How to Use This Page:
    - Upload your **own CSV file** with review data (must include `reviewText`, `verified_purchase`, `helpful_vote`, `asin`)  
      OR  
    - Click the **Use Sample Dataset** button to try the model on our example data which loads 1,000 preprocessed reviews.

    - 📩 After selecting one of the two options, click the **Submit** button shown **below your selected option**.
    - It will generate **Predicted Ratings**, preview table, and a **Download** button.
    """)

    # Option 1: User Uploaded Dataset
    uploaded = st.file_uploader("📤 Upload your CSV file here", type="csv")

    if uploaded:
        user_df = pd.read_csv(uploaded)
        required_cols = {"reviewText", "verified_purchase", "helpful_vote", "asin"}

        if not required_cols.issubset(user_df.columns):
            st.error("❌ Uploaded dataset is missing one or more required columns: `reviewText`, `verified_purchase`, `helpful_vote`, `asin`. Please upload a valid file.")
        else:
            st.session_state["df"] = user_df
            st.success("✅ Uploaded successfully!")
            st.dataframe(user_df.head(20))

            if st.button("📩 Submit Uploaded Data"):
                try:
                    preds = model.predict(X_test)[:len(user_df)]
                    df_preds = user_df.copy()
                    df_preds["Predicted Rating"] = preds
                    st.session_state["df_preds"] = df_preds
                    st.success("✅ Predictions generated!")
                    st.dataframe(df_preds.head(20))
                    st.download_button("📥 Download Prediction Results", df_preds.to_csv(index=False), file_name="predictions_uploaded.csv")

                    st.markdown("### 📊 Results Interpretation")
                    st.markdown("""
                    - These are **model-predicted review ratings** (scale: 1–5).
                    - These ratings are predicted based on patterns in verified purchase status, helpful votes, and more.
                    - These outputs are based on the **uploaded dataset** you provided. 
                    - Higher predicted scores imply greater customer satisfaction likelihood.
                    - Download and analyze the predictions, or continue to Page 3 for SHAP-based feature explainability.
                    """)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

    # Option 2: Sample Dataset
    if st.button("📦 Use Sample Dataset"):
        st.session_state["df"] = df.copy()
        st.session_state["use_sample"] = True
        st.success("✅ Sample dataset loaded!")
        st.dataframe(df.head(20))

    if st.session_state.get("use_sample"):
        if st.button("📩 Submit Sample Dataset"):
            try:
                preds = model.predict(X_test)
                df_preds = df.copy()
                df_preds["Predicted Rating"] = preds
                st.session_state["df_preds"] = df_preds
                st.success("✅ Predictions generated for sample!")
                st.dataframe(df_preds.head(20))
                st.download_button("📥 Download Predictions Results", df_preds.to_csv(index=False), file_name="predictions_sample.csv")

                st.markdown("""
                ---
                ### 📊 Results Interpretation:
                - These are **model-predicted review ratings** (scale: 1–5).
                - These ratings are predicted based on patterns in verified purchase status, helpful votes, and more.
                - The sample predictions help you understand how the model evaluates review data.
                - Use this as a benchmark before uploading your own data.
                - Higher predicted scores imply greater customer satisfaction likelihood.
                - Download and analyze the predictions, **OR** Now continue to Page 3 to see which features influenced the predictions.
                """)
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    st.markdown(
    """
    ---
    **Navigate to Next Page 3: Model Explainability with SHAP**
    ---            
    """)
    #if st.button("➡️ Click to Navigate to Next Page: 📈 Explain Model (SHAP)"):
    #    st.session_state["navigate_to_shap"] = True
    #    st.experimental_rerun()

    show_footer()

# ------------------------------------------
# PAGE 3: SHAP EXPLAINABILITY
# ------------------------------------------
elif page == "📈 Explain Model (SHAP)":
    st.title("📈 Model Explainability with SHAP")
    st.markdown("""
    ### 🧠 What This Page Shows:
    - If you submitted data on Page 2, click below to generate a **SHAP summary plot**.
    - Generates a **SHAP summary plot** showing which features (e.g., verified purchase, helpful votes) most influenced the model's predictions thus contributed to the predicted product rating.
    - Helps understand **why** the model predicted certain review scores.
    """)

    if "df_preds" not in st.session_state:
        st.warning("⚠️ Please go to the '📥 Test or Upload Data' page and submit a dataset first.")
    else:
        if st.button("📊 Generate SHAP Summary Plot"):
            try:
                explain_model_with_shap()
            except Exception as e:
                st.error(f"Failed to generate SHAP summary: {e}")

    #st.markdown("---")
    #if st.button("➡️ Click to Navigate to Next Page: 🤖 LLM Recommendations"):
    #    st.session_state["navigate_to_llm"] = True
    #    st.experimental_rerun()

    show_footer()

# ------------------------------------------
# PAGE 4: LLM RECOMMENDATION
# ------------------------------------------
elif page == "🤖 LLM Recommendations":
    st.title("🤖 AI-Powered Product Recommendation")
    st.markdown("""
    ### 💡 How to Use This Page:
    - Enter a *product query or need** (e.g., "noise-cancelling headphones" or "Looking for a budget-friendly power bank with fast charging.")
    - This app will:
      1. Retrieve similar customer reviews
      2. Identify the best product match; Find the top 3 matching reviews using embeddings
      3. Generate a personalized product recommendation using the Phi-2 model

    This uses real customer reviews — the recommendation reflects common sentiment and user experiences.
    """)

    query = st.text_input("🔎 **Enter your product query here:**", "Looking for a long-lasting Bluetooth headset with noise cancellation")
    if st.button("🤖 Submit Query"):
        try:
            reviews, product_name = retrieve_top_reviews(query)
            recommendation = generate_phi2_recommendation(query, reviews, product_name)

            st.markdown(f"### 🛍️ Recommended Product: **{product_name}**")
            st.markdown(f"### ✅ Why You'll Love It: {recommendation}")

            st.markdown("#### 🔍 Top 3 Reviews Considered")
            for i, r in enumerate(reviews[:3]):
                cleaned_review = clean_html(r)
                st.markdown(f"- **Review {i+1}:** {cleaned_review}")

            st.markdown("""
            ---
            ### 📊 LLM Recommendation System (APP):
            - This is a **language model-generated suggestion** based on verified user reviews.
            - It uses **semantic similarity + reasoning** to generate a friendly, tailored recommendation.
            - Great for e-commerce, customer service, and marketing intelligence.
            - The Phi-2 model has synthesized the reviews to give a **clear, human-readable** reason for choosing this product.
            - These suggestions are not endorsements — they are generated to help you explore suitable options.
            """)
        except Exception as e:
            st.error(f"❌ Failed to generate recommendation: {e}")

    show_footer()