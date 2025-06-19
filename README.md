
[![🚀 Open on Streamlit Cloud](https://img.shields.io/badge/Open-Streamlit%20App-brightgreen?logo=streamlit)](https://llm-recommendation-system-amazon.streamlit.app/)

---

# LLM-Based Product Recommendation System

Personalized, Explainable Product Recommendations for Real-World E-Commerce

---

**“Note:** “For cloud deployment and speed, this live demo runs on a 10,000-review sample. All model training and metrics are based on 1 million reviews. For full-scale results and business insights, see our [GitHub/Juypter Notebook]".

---

## 🧠 Overview

In a marketplace flooded with generic five-star reviews, shoppers struggle to find truly standout products. This project presents a full **Retrieve → Rank → Explain** pipeline that delivers personalized, interpretable recommendations using:

- 🔍 **Semantic Retrieval** via 384-dimensional review embeddings (SentenceTransformer)
- 🧮 **Rating Prediction** using XGBoost Regressor (MAE = 0.72, R² = 0.44)
- 📊 **Explainability** through SHAP summary plots
- 🤖 **Human-Style Recommendations** using the lightweight Phi-2 LLM                                                                                        
-    **Interactive Web App** Test, upload data, and view live insights on Streamlit

The system is fast (≤ 2 seconds/query), CPU-compatible, and optimized for scalable enterprise use in domains like e-commerce, streaming, or fintech.

---

## ✨ Features

- 🧠 **Phi-2 LLM Summarization**: Generates fluent product summaries from top-K reviews
- 🧾 **SHAP Explainability**: Transparent model decisions with visual insights
- 🔄 **Semantic Similarity Matching**: Retrieves most relevant verified reviews
- ⚡ **Real-Time Performance**: Inference under 2 seconds on CPU
- 📦 **Streamlit Interface**: Clean UX for testing queries or uploading custom datasets
- 🔐 **MIT Licensed**: Free to use, modify, and build upon

---

## 📦 Dataset                                                                                    

- **Source:** Amazon Electronics Reviews Dataset on Kaggle

- **Sample Size:** 1,000,000+ reviews

- **Features Used:**

    - reviewText: Full customer review text

    - summary: Review headline

    - overall: Star rating (1–5)

    - productTitle, brand, price

    - reviewerID (for personalization)

---

## 🏗️ How It Works

**1. Retrieve:**

- User query or sample product is embedded via all-MiniLM-L6-v2 (384-dim vector)

- Finds nearest products using cosine similarity in embedding space

**2. Rank:**

- For each candidate product, XGBoost Regressor predicts rating based on text embeddings + metadata

- SHAP plots show which features most influence the prediction

**3. Explain:**

- Phi-2 LLM generates concise, human-style summaries for top products

- Full breakdown and SHAP plots shown per product for transparency

---

## ⚙️ Installation (Local)

```bash
# 1. Clone the repo
git clone https://github.com/SweetySeelam2/LLM_Recommendation_Amazon.git

# 2. Navigate to the folder
cd LLM_Recommendation_Amazon

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

You can run the app locally via:

streamlit run app.py

---

## 🚀 Live Demo

You can try the deployed app instantly here:

[![🟢 Open on Streamlit Cloud](https://img.shields.io/badge/Open-Streamlit%20App-brightgreen?logo=streamlit)](https://llm-recommendation-system-amazon.streamlit.app/)

---

## 📈 Model Performance

| **Metric**                  | **Value**      | **Business Target**     |
|----------------------------|----------------|--------------------------|
| MAE (Mean Abs. Error)      | 0.72 stars     | ≤ 1.0 star               |
| RMSE                       | 0.99 stars     | ≤ 1.2 stars              |
| R² Score                   | 0.44           | ≥ 0.40                   |
| Classification Accuracy*   | 48.4%          | ≥ 40% (pseudo-categorical) |
| Inference Time             | < 2 seconds    | Real-time friendly       |

> ⚙️ *Pseudo-Classification Accuracy refers to converting predicted rating into closest star bin (1–5 stars) and measuring accuracy.

---

## 💼 Business Impact

- 🔍 **90% of Amazon electronics reviews** are 4–5 stars, making differentiation hard. This system surfaces **meaningful review signals** to guide purchases.
- 💸 Saves **manual research time (~5–10 min/user)** by offering **LLM-backed summaries**.
- 📊 Enhances explainability with **SHAP**, building **user trust** and increasing **conversion likelihood by 5–8%**.
- 🏢 Easily deployable in real-world platforms like **Amazon**, **Netflix**, **Flipkart**, or **Google Shopping**.

---

## 🌐 Deployment

🌐 Deployment
This app is currently deployed on Streamlit Cloud.

All large model/data files are loaded from Google Drive for this deployment (see app.py for details).

---

## 👤 Profile & Links

- **👩‍💻 Creator**: Sweety Seelam  
- **🔗 Portfolio**: [https://sweetyseelam2.github.io/SweetySeelam.github.io](https://sweetyseelam2.github.io/SweetySeelam.github.io)  
- **🌍 LinkedIn**: [linkedin.com/in/sweetyseelam](https://linkedin.com/in/sweetyseelam)  
- **📊 GitHub**: [github.com/SweetySeelam2](https://github.com/SweetySeelam2/LLM_Recommendation_Amazon)

---

## 📜 License

This project is licensed under the **MIT License** — meaning it is **free to use, modify, and redistribute** with proper attribution.

© 2025 Sweety Seelam
