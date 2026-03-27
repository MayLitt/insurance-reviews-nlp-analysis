# NLP Project 2 — Insurance Review Analysis Platform

## Project Overview

This project implements a complete **Natural Language Processing (NLP) pipeline** applied to insurance reviews.
It combines:

* Data exploration and preprocessing
* Unsupervised learning (topic modeling, embeddings)
* Supervised learning (classification models)
* Interactive applications using **Streamlit**

The final result is an **interactive platform** allowing users to:

* Predict sentiment, rating, and topic from a review
* Analyze insurers based on customer feedback
* Perform semantic search on reviews
* Ask questions using a **RAG (Retrieval-Augmented Generation) system**

---

## 📂 Project Structure

```
PROJET_2_NLP/
│
├── data/
│   ├── reviews_full.csv
│   ├── avis_avec_themes.csv
│   ├── avis_embeddings.npy
│   └── ...
│
├── models/
│   ├── model_note/
│   ├── model_sentiment/
│   └── model_theme/
│
├── src/
│   ├── app.py                # Main Streamlit app
│   ├── app_prediction.py    # Prediction module
│   ├── app_insurer.py       # Analysis dashboard
│   ├── exploration.ipynb    # Data exploration notebook
│   └── test.py
│
├── tensorboard/
│   ├── metadata.tsv
│   └── vectors.tsv
│
├── .streamlit/
│   └── secrets.toml         # API keys (Gemini)
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <repo_url>
cd PROJET_2_NLP
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 API Configuration (IMPORTANT)

This project uses **Google Gemini API** for the QA system.

Create a file:

```
.streamlit/secrets.toml
```

Add your API key:

```toml
GEMINI_API_KEY = "your_api_key_here"
```

⚠️ Without this, the **QA System will not work**.

---

## Run the Application

From the root folder:

```bash
streamlit run src/app.py
```

The app will open automatically in your browser.

---

## Features

### 1. Review Prediction

* Input a review (in English)
* Predict:

  * Sentiment (positive / neutral / negative)
  * Star rating (1–5)
  * Topic (pricing, claims, etc.)
* Display:

  * Confidence scores
  * Probability distributions
  * **SHAP explanations (word impact)**

---

### 2. Insurer Analysis Dashboard

* Global statistics:

  * Average ratings
  * Sentiment distribution
* Top insurers ranking
* Theme distribution
* Detailed insurer view with:

  * Metrics
  * Visualizations
  * Automatic summary generation

---

### 3. Review Search

* Keyword search
* **Semantic search using embeddings**
* Filtering by:

  * Insurer
  * Sentiment
  * Rating

---

### 4. QA System (RAG)

* Ask questions about insurers
* Uses:

  * Semantic search
  * Gemini LLM
* Returns:

  * Structured answer
  * Supporting reviews

---

## Models Used

All models are based on **DistilBERT fine-tuned**:

* Sentiment classification
* Star rating prediction
* Theme detection

Performance (Macro F1):

* Sentiment: **0.67**
* Rating: **0.47**
* Theme: **0.72**

---

## Data

* ~24,000 reviews
* 56 insurers
* Preprocessed text (translated & cleaned)

Main columns used:

* `note`
* `avis_nllb_en`
* `assureur`
* `theme`
* `sentiment` (derived)

---

## NLP Techniques Implemented

### Supervised Learning

* Transformer models (DistilBERT)
* Multi-task prediction (sentiment, rating, topic)

### Unsupervised Learning

* Sentence embeddings (SentenceTransformers)
* Semantic similarity (cosine similarity)

### Explainability

* SHAP values for model interpretation

### RAG (Retrieval-Augmented Generation)

* Semantic retrieval
* LLM (Gemini) for answer generation

---

## Notebook

The notebook `exploration.ipynb` contains:

* Data cleaning
* Exploratory analysis
* Feature engineering
* Model experimentation

---

## ⚠️ Important Notes

* The app requires:

  * pretrained models in `/models`
  * embeddings file in `/data`
* If files are missing → the app will not run correctly
* The QA system requires an internet connection

---

## Demo

A 5-minute demo video is provided to:

* explain the methodology
* demonstrate the application

---

## Author

Engineering student specializing in Data & AI.

---

## How to Test (Quick Guide for Professor)

1. Install dependencies
2. Add Gemini API key
3. Run:

```bash
streamlit run src/app.py
```

4. Test:

* Write a review → check predictions
* Explore dashboard
* Try semantic search
* Ask a question in QA system

---

## Summary

This project demonstrates a **complete NLP pipeline**, from raw data to an interactive AI-powered application, combining:

* Machine learning
* Deep learning
* Explainability
* Information retrieval
* Generative AI

---
