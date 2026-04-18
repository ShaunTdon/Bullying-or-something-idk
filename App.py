import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from transformers import pipeline
from fuzzy_engine import get_fuzzy_risk
from datetime import datetime
import time
import random
import torch # Needed now that fuzzy_engine uses torch

st.set_page_config(page_title="CyberGuard AI - ANFIS Monitor", layout="wide")

# --- 1. DATA LOADING (Replacing Mock Comments) ---
@st.cache_data
def load_real_data():
    try:
        # Loading your YouTube dataset
        df = pd.read_csv("youtoxic_english_1000.csv")
        return df['Text'].tolist()
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return ["Error: Could not load youtoxic_english_1000.csv"]

REAL_SAMPLES = load_real_data()

@st.cache_resource
def load_models():
    # Loading the "Perception" layer (BERT)
    t = pipeline("text-classification", model="unitary/toxic-bert")
    s = pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")
    return t, s

tox_p, sent_p = load_models()

if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=['Time', 'Risk', 'Text'])

# --- SIDEBAR CONTROLS ---
st.sidebar.title("🛡️ CyberGuard Control Panel")
start_demo = st.sidebar.toggle("🚀 Start Real-Time Monitoring", value=False)

if st.sidebar.button("🗑️ Clear Dashboard History"):
    st.session_state.history = pd.DataFrame(columns=['Time', 'Risk', 'Text'])
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("""
**Architecture: ANFIS**
- **Perception:** BERT NLP
- **Reasoning:** Neuro-Fuzzy Inference
- **Optimization:** Backpropagation
- **Dataset:** YouTube Toxic 1000
""")

st.title("🛡️ CyberGuard AI: ANFIS Risk Assessment")
st.caption("Hybrid Deep Learning & Fuzzy Logic Monitoring System")
placeholder = st.empty()

if start_demo:
    counter = 0 
    while True:
        counter += 1
        # Picking a random real comment from your CSV
        text = random.choice(REAL_SAMPLES)
        
        # 1. Perception Layer (BERT)
        t_res = tox_p(text[:512])[0] # Truncate to avoid BERT length errors
        t_val = t_res['score'] if t_res['label'] == 'toxic' else 1 - t_res['score']
        
        s_res = sent_p(text[:512])[0]
        s_val = 0.8 if s_res['label'] == "NEG" else 0.2 if s_res['label'] == "POS" else 0.5
        
        # 2. Reasoning Layer (ANFIS Weights from anfis_weights.pth)
        risk, f_tox, f_sent = get_fuzzy_risk(t_val, s_val)
        
        # Update History
        ts = datetime.now().strftime("%H:%M:%S")
        new_row = pd.DataFrame({'Time': [ts], 'Risk': [risk], 'Text': [text]})
        st.session_state.history = pd.concat([st.session_state.history, new_row], ignore_index=True).tail(15)

        with placeholder.container():
            col_left, col_right = st.columns([1, 2])
            
            with col_left:
                st.subheader("Live Analysis Feed")
                # 5-Level Risk Alert System
                if risk >= 0.8:
                    st.error(f"🚨 **CRITICAL RISK:** {text}")
                elif risk >= 0.6:
                    st.warning(f"⚠️ **ELEVATED RISK:** {text}")
                elif risk >= 0.4:
                    st.info(f"🔍 **MONITORING:** {text}")
                elif risk >= 0.2:
                    st.info(f"🟡 **CAUTION:** {text}")
                else:
                    st.success(f"✅ **SAFE:** {text}")
                
                st.caption(f"Optimized ANFIS Calculation: {risk:.4f}")
                st.caption(f"Source: YouTube Dataset Sample")
            
            with col_right:
                g1, g2, g3 = st.columns(3)
                
                def draw_g(n, v, c):
                    fig = go.Figure(go.Indicator(mode="gauge+number", value=v, title={'text': n},
                                  gauge={'axis':{'range':[0,1]}, 'steps':[{'range':[0,1], 'color':c}], 'bar':{'color':'white'}}))
                    fig.update_layout(height=250, margin=dict(t=50, b=0, l=10, r=10), paper_bgcolor='rgba(0,0,0,0)', font={'color':'white'})
                    return fig

                g1.plotly_chart(draw_g("Toxicity (BERT)", f_tox, "orange"), use_container_width=True, key=f"t_{counter}")
                g2.plotly_chart(draw_g("Sentiment (BERT)", f_sent, "blue"), use_container_width=True, key=f"s_{counter}")
                g3.plotly_chart(draw_g("Final Risk (ANFIS)", risk, "red"), use_container_width=True, key=f"r_{counter}")

            st.markdown("---")
            st.subheader("Risk Trend Analysis (Optimized Inference)")
            
            fig_trend = go.Figure(go.Scatter(x=st.session_state.history['Time'], 
                                            y=st.session_state.history['Risk'], 
                                            fill='tozeroy', mode='lines+markers', line_color='#00ccff'))
            fig_trend.update_layout(xaxis_title="Time", yaxis_title="Risk Score",
                                   yaxis=dict(range=[0, 1.1]), 
                                   height=350, paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
            
            st.plotly_chart(fig_trend, use_container_width=True, key=f"trend_{counter}")

        time.sleep(2.5) 
        if not start_demo: break
else:
    st.info("👈 Toggle 'Start Real-Time Monitoring' to begin the ANFIS evaluation loop.")