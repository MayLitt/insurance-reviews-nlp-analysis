import streamlit as st
import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import plotly.graph_objects as go
import shap
import streamlit.components.v1 as components
from transformers import pipeline
import matplotlib.pyplot as plt

# --- CONFIG ---
st.set_page_config(
    page_title="Insurance Review Analyzer",
    page_icon="Insurance",
    layout="wide"
)

# --- LOAD MODELS ---
@st.cache_resource
def load_model(task_name):
    path = f"./models/model_{task_name}"
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    with open(f"{path}/label_mapping.json") as f:
        label_mapping = json.load(f)
    return tokenizer, model, label_mapping

# --- PREDICTION FUNCTION ---
def predict(text, tokenizer, model, label_mapping):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0].numpy()
    pred_idx = str(np.argmax(probs))
    return {
        'prediction': label_mapping[pred_idx],
        'confidence': float(probs[int(pred_idx)]),
        'probabilities': {label_mapping[str(i)]: float(p) for i, p in enumerate(probs)}
    }

# --- INTERFACE ---
st.title("Insurance Review Analyzer")
st.markdown("**Predict sentiment, star rating and theme from any insurance review**")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This app uses **DistilBERT fine-tuned** on 24,104 French insurance reviews translated to English.
    
    **Model performance (Macro F1):**
    - Sentiment: **0.67**
    - Star rating: **0.47**
    - Theme: **0.72**
    
    **Model:** `distilbert-base-uncased`  
    **Training data:** 24,104 reviews  
    **Classes:**
    - Sentiment: negative / neutral / positive
    - Rating: 1 to 5 stars
    - Theme: pricing / claims / customer service / coverage / cancellation
    """)
    
    st.markdown("---")
    st.header("Example reviews")
    
    ex_positive = "Excellent insurance, very competitive price and outstanding customer service. I highly recommend!"
    ex_neutral = "The insurance is correct, nothing exceptional. Prices are average compared to competitors."
    ex_negative = "Terrible experience, they cancelled my contract without any notice. Avoid this insurer at all costs!"
    ex_claims = "After my car accident, the claims process was very slow. I waited 3 months for reimbursement."
    
    if st.button("Positive review"):
        st.session_state.input_text = ex_positive
    if st.button("Neutral review"):
        st.session_state.input_text = ex_neutral
    if st.button("Negative review"):
        st.session_state.input_text = ex_negative
    if st.button("Claims review"):
        st.session_state.input_text = ex_claims

# Input
st.subheader("Enter a review")
user_input = st.text_area(
    "Review text (in English):",
    value=st.session_state.get('input_text', ''),
    placeholder="e.g. The customer service was excellent, very fast response and helpful advisor.",
    height=150
)

analyze_btn = st.button("Analyze Review", type="primary", use_container_width=True)

if analyze_btn:
    if not user_input.strip():
        st.warning("Please enter a review first!")
    else:
        with st.spinner("Loading models and analyzing..."):

            # Load models
            tok_s, mod_s, lab_s = load_model("sentiment")
            tok_n, mod_n, lab_n = load_model("note")
            tok_t, mod_t, lab_t = load_model("theme")

            # Predictions
            res_s = predict(user_input, tok_s, mod_s, lab_s)
            res_n = predict(user_input, tok_n, mod_n, lab_n)
            res_t = predict(user_input, tok_t, mod_t, lab_t)

        st.markdown("---")
        st.subheader("Results")

        # --- MAIN METRICS ---
        col1, col2, col3 = st.columns(3)

        with col1:
            s = res_s['prediction']
            st.metric(
                label="Sentiment",
                value=s.capitalize(),
                delta=f"Confidence: {res_s['confidence']:.0%}"
            )

        with col2:
            note_val = res_n['prediction']
            st.metric(
                label="Predicted Rating",
                value=f"{note_val} stars",
                delta=f"Confidence: {res_n['confidence']:.0%}"
            )

        with col3:
            t = res_t['prediction']
            st.metric(
                label="Theme",
                value=t.title(),
                delta=f"Confidence: {res_t['confidence']:.0%}"
            )

        st.markdown("---")

        # --- PROBABILITY CHARTS ---
        st.subheader("Prediction Probabilities")
        col1, col2, col3 = st.columns(3)

        def make_bar_chart(probs_dict, title, color):
            labels = list(probs_dict.keys())
            values = list(probs_dict.values())
            fig = go.Figure(go.Bar(
                x=values, y=labels,
                orientation='h',
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

        with col1:
            fig = make_bar_chart(res_s['probabilities'], "Sentiment", "#3498db")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = make_bar_chart(res_n['probabilities'], "Star Rating", "#f39c12")
            st.plotly_chart(fig, use_container_width=True)

        with col3:
            fig = make_bar_chart(res_t['probabilities'], "Theme", "#2ecc71")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # --- MODEL EXPLANATION (SHAP) ---
        st.subheader("Explanation")
        
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
                
                # Extract words and their impact scores
                words = shap_values[0].data
                impacts = shap_values[0].values[:, pred_idx]
                
                # Keep top 10 most important words
                word_impacts = [(w, i) for w, i in zip(words, impacts) if str(w).strip()]
                word_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
                top_10 = word_impacts[:10]
                top_10.reverse()
                
                top_words = [str(x[0]) for x in top_10]
                top_scores = [float(x[1]) for x in top_10]
                
                # Color coding based on sentiment contribution
                if res_s['prediction'] == 'positive':
                    colors = ['#00cc66' if s > 0 else '#ff0051' for s in top_scores]
                elif res_s['prediction'] == 'negative':
                    colors = ['#ff0051' if s > 0 else '#00cc66' for s in top_scores]
                else:
                    colors = ['#3498db' if s > 0 else '#e74c3c' for s in top_scores]
                
                # Plot SHAP values
                fig = go.Figure(go.Bar(
                    x=top_scores,
                    y=top_words,
                    orientation='h',
                    marker_color=colors,
                    text=[f"{s:.2f}" for s in top_scores],
                    textposition='outside'
                ))
                
                fig.update_layout(
                    title=f"Words driving the prediction: {res_s['prediction'].upper()}",
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                    xaxis_title="Impact on Decision",
                    yaxis_title="Words in text"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Could not generate SHAP explanation: {e}")

        # --- FINAL SUMMARY ---
        st.info(f"""
        **Summary:** This review is **{res_s['prediction']}** 
        with a predicted rating of **{res_n['prediction']} stars** 
        about **{res_t['prediction']}**.  
        The model is **{res_s['confidence']:.0%}** confident about the sentiment.
        """)