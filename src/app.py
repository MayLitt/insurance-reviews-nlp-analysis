import streamlit as st
import pandas as pd
import numpy as np
import torch
import json
import requests
import plotly.express as px
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm
import shap
import google.generativeai as genai

# --- CONFIG ---
st.set_page_config(
    page_title="Insurance Review Platform",
    page_icon="Insurance",
    layout="wide"
)

# ============================================================
# LOAD MODELS AND DATA
# ============================================================

@st.cache_resource
def load_prediction_model(task_name):
    path = f"./models/model_{task_name}"
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    with open(f"{path}/label_mapping.json") as f:
        label_mapping = json.load(f)
    return tokenizer, model, label_mapping

@st.cache_data
def load_data():
    df = pd.read_csv("./data/avis_avec_themes.csv", on_bad_lines='skip')
    df = df.dropna(subset=['note', 'avis_nllb_en']).reset_index(drop=True)
    df['note'] = pd.to_numeric(df['note'], errors='coerce')
    df = df.dropna(subset=['note'])

    # Convert rating to sentiment label
    def note_to_sentiment(note):
        if note <= 2: return 'negative'
        elif note == 3: return 'neutral'
        else: return 'positive'

    df['sentiment'] = df['note'].apply(note_to_sentiment)
    return df

@st.cache_resource
def load_embeddings_and_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = np.load("./data/avis_embeddings.npy")
    return model, embeddings

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def predict(text, tokenizer, model, label_mapping):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0].numpy()
    pred_idx = str(np.argmax(probs))
    return {
        'prediction': label_mapping[pred_idx],
        'confidence': float(probs[int(pred_idx)]),
        'probabilities': {label_mapping[str(i)]: float(p) for i, p in enumerate(probs)}
    }

def semantic_search(query, model, embeddings, df, top_k=10):
    query_vec = model.encode([query])[0]

    # Compute cosine similarity with all embeddings
    scores = []
    for i, emb in enumerate(embeddings[:len(df)]):
        score = np.dot(query_vec, emb) / (norm(query_vec) * norm(emb))
        scores.append((i, score))

    # Sort results by similarity score
    scores.sort(key=lambda x: x[1], reverse=True)

    results = []
    for idx, score in scores[:top_k]:
        row = df.iloc[idx]
        results.append({
            'score': score,
            'assureur': row['assureur'],
            'note': row['note'],
            'sentiment': row['sentiment'],
            'theme': row.get('theme', 'unknown'),
            'avis': str(row['avis_nllb_en'])[:300]
        })
    return results

def generate_summary(df_insurer, insurer_name):
    total = len(df_insurer)
    avg_note = df_insurer['note'].mean()
    sentiment_counts = df_insurer['sentiment'].value_counts()

    # Extract most frequent theme if available
    if 'theme' in df_insurer.columns and len(df_insurer['theme'].dropna()) > 0:
        theme_counts = df_insurer['theme'].value_counts()
        top_theme = theme_counts.index[0]
    else:
        top_theme = "unknown"

    pos_pct = sentiment_counts.get('positive', 0) / total * 100
    neg_pct = sentiment_counts.get('negative', 0) / total * 100
    neu_pct = sentiment_counts.get('neutral', 0) / total * 100

    overall = "very well rated" if avg_note >= 4 else ("moderately rated" if avg_note >= 3 else "poorly rated")

    return f"""
    **{insurer_name}** is {overall} with an average rating of **{avg_note:.1f}/5** 
    based on **{total} reviews**.
    - **{pos_pct:.0f}%** of customers are satisfied (positive sentiment)
    - **{neg_pct:.0f}%** of customers are dissatisfied (negative sentiment)
    - **{neu_pct:.0f}%** of customers are neutral
    The most discussed topic is **{top_theme}**.
    """

def generate_rag_answer(question, context_text):
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
    except KeyError:
        return "Gemini API key not found in secrets."

    # Prompt for RAG-based QA
    prompt = f"""You are an expert insurance analyst. Answer the question accurately using ONLY the context below.
RULES:
1. Quote exact insurer names from the context (inside brackets like [Insurer Name]).
2. Give precise facts, not general summaries.
3. If the context doesn't answer the question, say so clearly.

CONTEXT:
{context_text}

QUESTION: {question}"""

    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"API Error: {e}"

def make_bar_chart(probs_dict, title, color):
    labels = list(probs_dict.keys())
    values = list(probs_dict.values())

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation='h',
        marker_color=color,
        text=[f"{v:.0%}" for v in values],
        textposition='outside'
    ))

    fig.update_layout(
        title=title,
        xaxis=dict(range=[0, 1], tickformat='.0%'),
        height=200,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig

# ============================================================
# MAIN NAVIGATION
# ============================================================

df = load_data()
insurers = sorted(df['assureur'].unique().tolist())

with st.sidebar:
    st.title("Insurance Review Platform")
    st.markdown("---")

    main_page = st.radio(
        "Select a tool:",
        ["Review Prediction", "Insurer Analysis"]
    )

    st.markdown("---")

    st.markdown("""
    **Models (Macro F1):**
    - Sentiment: 0.67
    - Star rating: 0.47
    - Theme: 0.72
    
    **Model:** DistilBERT fine-tuned  
    **Data:** 24,104 reviews  
    **Insurers:** 56
    """)

# ============================================================
# APP 1 : REVIEW PREDICTION
# ============================================================

if main_page == "Review Prediction":
    st.title("Review Prediction")
    st.markdown("Predict sentiment, star rating and theme from any insurance review")
    st.markdown("---")

    col_input, col_examples = st.columns([3, 1])

    with col_examples:
        st.markdown("**Examples:**")
        ex_positive = "Excellent insurance, very competitive price and outstanding customer service. I highly recommend!"
        ex_neutral = "The insurance is correct, nothing exceptional. Prices are average compared to competitors."
        ex_negative = "Terrible experience, they cancelled my contract without any notice. Avoid this insurer!"
        ex_claims = "After my car accident, the claims process was very slow. I waited 3 months for reimbursement."
        if st.button("Positive review"):
            st.session_state.input_text = ex_positive
        if st.button("Neutral review"):
            st.session_state.input_text = ex_neutral
        if st.button("Negative review"):
            st.session_state.input_text = ex_negative
        if st.button("Claims review"):
            st.session_state.input_text = ex_claims

    with col_input:
        user_input = st.text_area(
            "Review text (in English):",
            value=st.session_state.get('input_text', ''),
            placeholder="e.g. The customer service was excellent, very fast and helpful.",
            height=150
        )

    if st.button("Analyze Review", type="primary", use_container_width=True):
        if not user_input.strip():
            st.warning("Please enter a review first!")
        else:
            with st.spinner("Analyzing..."):
                tok_s, mod_s, lab_s = load_prediction_model("sentiment")
                tok_n, mod_n, lab_n = load_prediction_model("note")
                tok_t, mod_t, lab_t = load_prediction_model("theme")
                res_s = predict(user_input, tok_s, mod_s, lab_s)
                res_n = predict(user_input, tok_n, mod_n, lab_n)
                res_t = predict(user_input, tok_t, mod_t, lab_t)

            st.markdown("---")
            st.subheader("Results")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Sentiment", res_s['prediction'].capitalize(), delta=f"Confidence: {res_s['confidence']:.0%}")
            with col2:
                st.metric("Predicted Rating", f"{res_n['prediction']} stars", delta=f"Confidence: {res_n['confidence']:.0%}")
            with col3:
                st.metric("Theme", res_t['prediction'].title(), delta=f"Confidence: {res_t['confidence']:.0%}")

            st.markdown("---")
            st.subheader("Prediction Probabilities")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.plotly_chart(make_bar_chart(res_s['probabilities'], "Sentiment", "#3498db"), use_container_width=True)
            with col2:
                st.plotly_chart(make_bar_chart(res_n['probabilities'], "Star Rating", "#f39c12"), use_container_width=True)
            with col3:
                st.plotly_chart(make_bar_chart(res_t['probabilities'], "Theme", "#2ecc71"), use_container_width=True)

            st.markdown("---")
            st.subheader("Explanation — Word Impact (SHAP)")
            with st.spinner("Calculating word impact..."):
                try:
                    def predict_for_shap(texts):
                        safe_texts = [str(t) for t in texts]
                        inputs = tok_s(safe_texts, return_tensors="pt", truncation=True, max_length=128, padding=True)
                        inputs = {k: v.to(torch.long) for k, v in inputs.items()}
                        with torch.no_grad():
                            outputs = mod_s(**inputs)
                        return torch.softmax(outputs.logits, dim=-1).numpy()

                    explainer = shap.Explainer(predict_for_shap, tok_s)
                    shap_values = explainer([user_input])
                    labels = list(res_s['probabilities'].keys())
                    pred_idx = labels.index(res_s['prediction'])
                    words = shap_values[0].data
                    impacts = shap_values[0].values[:, pred_idx]
                    word_impacts = [(w, i) for w, i in zip(words, impacts) if str(w).strip()]
                    word_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
                    top_10 = word_impacts[:10]
                    top_10.reverse()
                    top_words = [str(x[0]) for x in top_10]
                    top_scores = [float(x[1]) for x in top_10]
                    if res_s['prediction'] == 'positive':
                        colors = ['#00cc66' if s > 0 else '#ff0051' for s in top_scores]
                    elif res_s['prediction'] == 'negative':
                        colors = ['#ff0051' if s > 0 else '#00cc66' for s in top_scores]
                    else:
                        colors = ['#3498db' if s > 0 else '#e74c3c' for s in top_scores]
                    fig = go.Figure(go.Bar(
                        x=top_scores, y=top_words, orientation='h',
                        marker_color=colors,
                        text=[f"{s:.2f}" for s in top_scores],
                        textposition='outside'
                    ))
                    fig.update_layout(
                        title=f"Words driving the prediction: {res_s['prediction'].upper()}",
                        height=400,
                        xaxis_title="Impact on Decision",
                        yaxis_title="Words"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not generate SHAP explanation: {e}")

            st.info(f"""
            **Summary:** This review is **{res_s['prediction']}** 
            with a predicted rating of **{res_n['prediction']} stars** 
            about **{res_t['prediction']}**.
            The model is **{res_s['confidence']:.0%}** confident about the sentiment.
            """)

# ============================================================
# APP 2 : INSURER ANALYSIS
# ============================================================

elif main_page == "Insurer Analysis":
    st.title("Insurer Analysis Dashboard")
    st.markdown("Explore and analyze insurance reviews by insurer")
    st.markdown("---")

    page = st.radio(
        "Section:",
        ["Overview", "Insurer Details", "Review Search", "QA System"],
        horizontal=True
    )

    # --- OVERVIEW ---
    if page == "Overview":
        st.header("Overview — All Insurers")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Reviews", f"{len(df):,}")
        with col2:
            st.metric("Number of Insurers", df['assureur'].nunique())
        with col3:
            st.metric("Average Rating", f"{df['note'].mean():.2f}/5")
        with col4:
            pos_pct = (df['sentiment'] == 'positive').mean() * 100
            st.metric("Positive Reviews", f"{pos_pct:.1f}%")

        st.markdown("---")
        st.subheader("Top 10 Insurers by Average Rating")
        top_insurers = (
            df.groupby('assureur')
            .agg(avg_note=('note', 'mean'), count=('note', 'count'))
            .reset_index()
            .query('count >= 30')
            .sort_values('avg_note', ascending=False)
            .head(10)
        )
        fig = px.bar(
            top_insurers, x='assureur', y='avg_note',
            color='avg_note', color_continuous_scale='RdYlGn',
            text='avg_note', labels={'assureur': 'Insurer', 'avg_note': 'Average Rating'}
        )
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sentiment Distribution")
            sentiment_counts = df['sentiment'].value_counts()
            fig = px.pie(
                values=sentiment_counts.values, names=sentiment_counts.index,
                color=sentiment_counts.index,
                color_discrete_map={'positive': '#2ecc71', 'neutral': '#f39c12', 'negative': '#e74c3c'}
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Theme Distribution")
            if 'theme' in df.columns:
                theme_counts = df['theme'].value_counts()
                fig = px.bar(
                    x=theme_counts.values, y=theme_counts.index,
                    orientation='h', color=theme_counts.values,
                    color_continuous_scale='Blues',
                    labels={'x': 'Count', 'y': 'Theme'}
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("All Insurers — Summary Table")
        summary_table = (
            df.groupby('assureur')
            .agg(
                avg_rating=('note', 'mean'),
                total_reviews=('note', 'count'),
                pct_positive=('sentiment', lambda x: (x == 'positive').mean() * 100),
                pct_negative=('sentiment', lambda x: (x == 'negative').mean() * 100)
            )
            .reset_index()
            .sort_values('avg_rating', ascending=False)
            .round(2)
        )
        summary_table.columns = ['Insurer', 'Avg Rating', 'Total Reviews', '% Positive', '% Negative']
        st.dataframe(summary_table, use_container_width=True, height=400)

    # --- INSURER DETAILS ---
    elif page == "Insurer Details":
        st.header("Insurer Details")
        selected_insurer = st.selectbox("Select an insurer:", insurers)
        df_ins = df[df['assureur'] == selected_insurer]

        st.markdown("---")
        st.subheader("Automatic Summary")
        st.markdown(generate_summary(df_ins, selected_insurer))
        st.markdown("---")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Reviews", len(df_ins))
        with col2:
            st.metric("Average Rating", f"{df_ins['note'].mean():.2f}/5")
        with col3:
            pos = (df_ins['sentiment'] == 'positive').mean() * 100
            st.metric("Positive Reviews", f"{pos:.1f}%")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Rating Distribution")
            note_counts = df_ins['note'].value_counts().sort_index()
            fig = px.bar(
                x=note_counts.index, y=note_counts.values,
                labels={'x': 'Stars', 'y': 'Count'},
                color=note_counts.index, color_continuous_scale='RdYlGn'
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Sentiment Distribution")
            sent_counts = df_ins['sentiment'].value_counts()
            fig = px.pie(
                values=sent_counts.values, names=sent_counts.index,
                color=sent_counts.index,
                color_discrete_map={'positive': '#2ecc71', 'neutral': '#f39c12', 'negative': '#e74c3c'}
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        if 'theme' in df.columns:
            st.subheader("Theme Breakdown")
            theme_sent = df_ins.groupby(['theme', 'sentiment']).size().reset_index(name='count')
            fig = px.bar(
                theme_sent, x='theme', y='count', color='sentiment',
                color_discrete_map={'positive': '#2ecc71', 'neutral': '#f39c12', 'negative': '#e74c3c'},
                barmode='stack'
            )
            fig.update_layout(height=350, xaxis_tickangle=-20)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Recent Reviews")
        n_reviews = st.slider("Number of reviews to display:", 5, 50, 10)
        cols_to_show = ['note', 'sentiment', 'theme', 'avis_nllb_en'] if 'theme' in df.columns else ['note', 'sentiment', 'avis_nllb_en']
        st.dataframe(df_ins[cols_to_show].head(n_reviews), use_container_width=True)

    # --- REVIEW SEARCH ---
    elif page == "Review Search":
        st.header("Review Search")
        search_type = st.radio("Search type:", ["Keyword Search", "Semantic Search"], horizontal=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            filter_insurer = st.selectbox("Filter by insurer:", ["All"] + insurers)
        with col2:
            filter_sentiment = st.selectbox("Filter by sentiment:", ["All", "positive", "neutral", "negative"])
        with col3:
            filter_note = st.selectbox("Filter by rating:", ["All", "1.0", "2.0", "3.0", "4.0", "5.0"])

        df_filtered = df.copy()
        if filter_insurer != "All":
            df_filtered = df_filtered[df_filtered['assureur'] == filter_insurer]
        if filter_sentiment != "All":
            df_filtered = df_filtered[df_filtered['sentiment'] == filter_sentiment]
        if filter_note != "All":
            df_filtered = df_filtered[df_filtered['note'] == float(filter_note)]

        st.markdown(f"**{len(df_filtered):,} reviews match your filters**")
        st.markdown("---")

        if search_type == "Keyword Search":
            query = st.text_input("Search keyword:", placeholder="e.g. reimbursement, cancel, price...")
            if query:
                results = df_filtered[df_filtered['avis_nllb_en'].str.contains(query, case=False, na=False)]
                st.markdown(f"**{len(results)} results found for '{query}'**")
                cols_to_show = ['assureur', 'note', 'sentiment', 'theme', 'avis_nllb_en'] if 'theme' in df.columns else ['assureur', 'note', 'sentiment', 'avis_nllb_en']
                st.dataframe(results[cols_to_show].head(50), use_container_width=True)
        else:
            query = st.text_input("Semantic query:", placeholder="e.g. very long waiting time for reimbursement...")
            top_k = st.slider("Number of results:", 5, 20, 10)
            if query and st.button("Search", type="primary"):
                with st.spinner("Running semantic search..."):
                    model_search, embeddings = load_embeddings_and_model()
                    results = semantic_search(query, model_search, embeddings, df_filtered, top_k)
                for i, res in enumerate(results):
                    with st.expander(f"Result {i+1} — {res['assureur']} | {res['note']} stars | Score: {res['score']:.3f}"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"**Insurer:** {res['assureur']}")
                        with col2:
                            st.write(f"**Sentiment:** {res['sentiment']}")
                        with col3:
                            st.write(f"**Theme:** {res['theme']}")
                        st.write(f"**Review:** {res['avis']}")

    # --- QA SYSTEM ---
    elif page == "QA System":
        st.header("QA System — Ask questions about insurers")
        st.markdown("Ask any question about insurance reviews and get an answer powered by Gemini AI.")

        selected_insurer_qa = st.selectbox("Focus on a specific insurer (optional):", ["All insurers"] + insurers)
        question = st.text_input(
            "Your question:",
            placeholder="e.g. What do customers say about the claims process at Direct Assurance?"
        )

        if question and st.button("Get Answer", type="primary"):
            with st.spinner("Searching relevant reviews..."):
                model_search, embeddings = load_embeddings_and_model()
                df_qa = df.copy()
                if selected_insurer_qa != "All insurers":
                    df_qa = df_qa[df_qa['assureur'] == selected_insurer_qa]
                results = semantic_search(question, model_search, embeddings, df_qa, top_k=5)

            context_reviews = "\n".join([
                f"- [{r['assureur']} | {r['note']} stars | {r['sentiment']}]: {r['avis']}"
                for r in results
            ])

            st.markdown("---")
            st.subheader("Answer")
            with st.spinner("Generating answer with Gemini..."):
                answer = generate_rag_answer(question, context_reviews)
            st.success(answer)

            st.markdown("---")
            st.subheader("Most relevant reviews")
            for i, res in enumerate(results):
                with st.expander(f"Review {i+1} — {res['assureur']} | {res['note']} stars | Relevance: {res['score']:.3f}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**Insurer:** {res['assureur']}")
                    with col2:
                        st.write(f"**Sentiment:** {res['sentiment']}")
                    with col3:
                        st.write(f"**Theme:** {res['theme']}")
                    st.write(f"**Review:** {res['avis']}")
