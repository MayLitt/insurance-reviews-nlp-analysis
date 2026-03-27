import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm
import json
import google.generativeai as genai

# --- CONFIG ---
st.set_page_config(
    page_title="Insurer Analysis",
    page_icon="Insurance",
    layout="wide"
)

# --- LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv(
        "./data/avis_avec_themes.csv",
        on_bad_lines='skip'
    )
    df = df.dropna(subset=['note', 'avis_nllb_en']).reset_index(drop=True)
    df['note'] = pd.to_numeric(df['note'], errors='coerce')
    df = df.dropna(subset=['note'])

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

# --- GENERATE RAG ANSWER ---
def generate_rag_answer(question, context_text):
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
    except KeyError:
        return "⚠️ Error: Gemini API key not found."

    prompt = f"""
    You are an expert data analyst in the insurance sector. 
    Your task is to answer the user's question accurately, USING ONLY the context provided below.
    
    STRICT RULES:
    1. If the question asks "Who", "Which insurer", etc., YOU MUST QUOTE THE EXACT NAMES of the insurers from the context (they are inside brackets like [Insurer Name | Stars]).
    2. Never write a general summary. Give precise facts based on the reviews.
    3. Briefly quote parts of the reviews to justify your answer.
    4. If no insurer name is found in the context to answer the question, simply state that the current data does not allow you to answer.
    
    CONTEXT (Customer reviews):
    {context_text}
    
    QUESTION:
    {question}
    """
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error during API call: {e}"
    

# --- SEMANTIC SEARCH ---
def semantic_search(query, model, embeddings, df, top_k=10):
    query_vec = model.encode([query])[0]
    scores = []
    for i, emb in enumerate(embeddings[:len(df)]):
        score = np.dot(query_vec, emb) / (norm(query_vec) * norm(emb))
        scores.append((i, score))
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

# --- INSURER SUMMARY ---
def generate_summary(df_insurer, insurer_name):
    total = len(df_insurer)
    avg_note = df_insurer['note'].mean()
    sentiment_counts = df_insurer['sentiment'].value_counts()
    
    # Handle missing or empty theme column
    if 'theme' in df_insurer.columns and len(df_insurer['theme'].dropna()) > 0:
        theme_counts = df_insurer['theme'].value_counts()
        top_theme = theme_counts.index[0]
    else:
        top_theme = "unknown"

    pos_pct = sentiment_counts.get('positive', 0) / total * 100
    neg_pct = sentiment_counts.get('negative', 0) / total * 100
    neu_pct = sentiment_counts.get('neutral', 0) / total * 100

    if avg_note >= 4:
        overall = "very well rated"
    elif avg_note >= 3:
        overall = "moderately rated"
    else:
        overall = "poorly rated"

    summary = f"""
    **{insurer_name}** is {overall} with an average rating of **{avg_note:.1f}/5** 
    based on **{total} reviews**.
    
    - **{pos_pct:.0f}%** of customers are satisfied (positive sentiment)
    - **{neg_pct:.0f}%** of customers are dissatisfied (negative sentiment)  
    - **{neu_pct:.0f}%** of customers are neutral
    
    The most discussed topic is **{top_theme}**.
    """
    return summary

# --- INTERFACE ---
df = load_data()

st.title("Insurer Analysis Dashboard")
st.markdown("Explore and analyze insurance reviews by insurer")
st.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Insurer Details", "Review Search", "QA System"]
)

insurers = sorted(df['assureur'].unique().tolist())

# ============================================================
# PAGE 1 : OVERVIEW
# ============================================================
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

    # Top 10 insurers by average rating
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
        top_insurers,
        x='assureur',
        y='avg_note',
        color='avg_note',
        color_continuous_scale='RdYlGn',
        text='avg_note',
        labels={'assureur': 'Insurer', 'avg_note': 'Average Rating'}
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(height=400, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Global sentiment distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentiment Distribution")
        sentiment_counts = df['sentiment'].value_counts()
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            color=sentiment_counts.index,
            color_discrete_map={
                'positive': '#2ecc71',
                'neutral': '#f39c12',
                'negative': '#e74c3c'
            }
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Theme Distribution")
        if 'theme' in df.columns:
            theme_counts = df['theme'].value_counts()
            fig = px.bar(
                top_insurers,
                x='assureur',
                y='avg_note',
                color='avg_note',
                color_continuous_scale='RdYlGn',
                text='avg_note',
                labels={'assureur': 'Insurer', 'avg_note': 'Average Rating'}
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Full summary table
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

# ============================================================
# PAGE 2 : INSURER DETAILS
# ============================================================
elif page == "Insurer Details":
    st.header("Insurer Details")

    selected_insurer = st.selectbox("Select an insurer:", insurers)
    df_ins = df[df['assureur'] == selected_insurer]

    st.markdown("---")

    # Résumé automatique
    st.subheader("Automatic Summary")
    summary = generate_summary(df_ins, selected_insurer)
    st.markdown(summary)

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
            x=note_counts.index,
            y=note_counts.values,
            labels={'x': 'Stars', 'y': 'Count'},
            color=note_counts.index,
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Sentiment Distribution")
        sent_counts = df_ins['sentiment'].value_counts()
        fig = px.pie(
            values=sent_counts.values,
            names=sent_counts.index,
            color=sent_counts.index,
            color_discrete_map={
                'positive': '#2ecc71',
                'neutral': '#f39c12',
                'negative': '#e74c3c'
            }
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Themes par assureur
    if 'theme' in df.columns:
        st.subheader("Theme Breakdown")
        theme_sent = df_ins.groupby(['theme', 'sentiment']).size().reset_index(name='count')
        fig = px.bar(
            theme_sent,
            x='theme',
            y='count',
            color='sentiment',
            color_discrete_map={
                'positive': '#2ecc71',
                'neutral': '#f39c12',
                'negative': '#e74c3c'
            },
            barmode='stack'
        )
        fig.update_layout(height=350, xaxis_tickangle=-20)
        st.plotly_chart(fig, use_container_width=True)

    # Avis récents
    st.subheader("Recent Reviews")
    n_reviews = st.slider("Number of reviews to display:", 5, 50, 10)
    cols_to_show = ['note', 'sentiment', 'theme', 'avis_nllb_en'] if 'theme' in df.columns else ['note', 'sentiment', 'avis_nllb_en']
    st.dataframe(
        df_ins[cols_to_show].head(n_reviews),
        use_container_width=True
    )

# ============================================================
# PAGE 3 : REVIEW SEARCH
# ============================================================
elif page == "Review Search":
    st.header("Review Search")

    search_type = st.radio(
        "Search type:",
        ["Keyword Search", "Semantic Search"],
        horizontal=True
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        filter_insurer = st.selectbox("Filter by insurer:", ["All"] + insurers)
    with col2:
        filter_sentiment = st.selectbox("Filter by sentiment:", ["All", "positive", "neutral", "negative"])
    with col3:
        filter_note = st.selectbox("Filter by rating:", ["All", "1.0", "2.0", "3.0", "4.0", "5.0"])

    # Appliquer les filtres
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
            results = df_filtered[
                df_filtered['avis_nllb_en'].str.contains(query, case=False, na=False)
            ]
            st.markdown(f"**{len(results)} results found for '{query}'**")
            cols_to_show = ['assureur', 'note', 'sentiment', 'theme', 'avis_nllb_en'] if 'theme' in df.columns else ['assureur', 'note', 'sentiment', 'avis_nllb_en']
            st.dataframe(results[cols_to_show].head(50), use_container_width=True)

    else:  # Semantic Search
        query = st.text_input(
            "Semantic query:",
            placeholder="e.g. very long waiting time for reimbursement..."
        )
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

# ============================================================
# PAGE 4 : QA SYSTEM (RAG)
# ============================================================
elif page == "QA System":
    st.header("QA System — Ask questions about insurers")
    st.markdown("Ask any question about insurance reviews and get an answer based on relevant reviews.")

    selected_insurer_qa = st.selectbox(
        "Focus on a specific insurer (optional):",
        ["All insurers"] + insurers
    )

    question = st.text_input(
        "Your question:",
        placeholder="e.g. What do customers say about the claims process at Direct Assurance?"
    )

    if question and st.button("Get Answer", type="primary"):
        with st.spinner("Searching relevant reviews..."):
            model_search, embeddings = load_embeddings_and_model()

            # Filtrer par assureur si sélectionné
            df_qa = df.copy()
            if selected_insurer_qa != "All insurers":
                df_qa = df_qa[df_qa['assureur'] == selected_insurer_qa]

            # Recherche sémantique
            results = semantic_search(question, model_search, embeddings, df_qa, top_k=5)

        st.markdown("---")
        st.subheader("Answer based on relevant reviews")

        # Construire le contexte
        context_reviews = "\n".join([
            f"- [{r['assureur']} | {r['note']} stars | {r['sentiment']}]: {r['avis']}"
            for r in results
        ])

        # Analyse simple basée sur les reviews trouvées
        sentiments = [r['sentiment'] for r in results]
        avg_score = np.mean([r['score'] for r in results])
        pos_count = sentiments.count('positive')
        neg_count = sentiments.count('negative')

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