import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from transformers import pipeline
from fuzzy_engine import get_fuzzy_risk
from datetime import datetime
import time
import random

st.set_page_config(page_title="CyberGuard AI - Infinite Monitor", layout="wide")

# 1. Mock Data
MOCK_COMMENTS = [
    "I hope you have a wonderful day!", "You are absolutely worthless.",
    "That is a very interesting point.", "Go jump off a bridge, you idiot.",
    "I totally disagree, but I respect you.", "Stop posting, you're annoying.",
    "This community is so helpful!", "You're a disgusting human. Leave.",
    "Wow, great job on the project!", "Kill yourself, no one wants you.",
    "Let's work together on this.", "You are the worst person I've met."
]

if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=['Time', 'Risk', 'Text'])

@st.cache_resource
def load_models():
    t = pipeline("text-classification", model="unitary/toxic-bert")
    s = pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")
    return t, s

tox_p, sent_p = load_models()

# --- SIDEBAR CONTROLS ---
st.sidebar.title("Simulation Controls")
start_demo = st.sidebar.toggle("🚀 Start Infinite Monitoring", value=False)
if st.sidebar.button("🗑️ Clear Graph History"):
    st.session_state.history = pd.DataFrame(columns=['Time', 'Risk', 'Text'])
    st.rerun()

st.title("🛡️ CyberGuard AI: Continuous Risk Assessment")
placeholder = st.empty()

if start_demo:
    counter = 0 
    while True:
        counter += 1
        text = random.choice(MOCK_COMMENTS)
        
        # Analysis
        t_res = tox_p(text)[0]
        t_val = t_res['score'] if t_res['label'] == 'toxic' else 1 - t_res['score']
        s_res = sent_p(text)[0]
        s_val = 0.8 if s_res['label'] == "NEG" else 0.2 if s_res['label'] == "POS" else 0.5
        
        # Fuzzy Engine
        risk, f_tox, f_sent = get_fuzzy_risk(t_val, s_val)
        
        # Update Data
        ts = datetime.now().strftime("%H:%M:%S")
        new_row = pd.DataFrame({'Time': [ts], 'Risk': [risk], 'Text': [text]})
        st.session_state.history = pd.concat([st.session_state.history, new_row], ignore_index=True).tail(15)

        with placeholder.container():
            col_left, col_right = st.columns([1, 2])
            
            with col_left:
                st.subheader("Live Feed")
                if risk > 0.7: st.error(f"**CRITICAL:** {text}")
                elif risk > 0.4: st.warning(f"**WARNING:** {text}")
                else: st.success(f"**SAFE:** {text}")
                st.caption(f"Last update: {ts}")
            
            with col_right:
                g1, g2, g3 = st.columns(3)
                
                def draw_g(n, v, c, k):
                    fig = go.Figure(go.Indicator(mode="gauge+number", value=v, title={'text': n},
                                  gauge={'axis':{'range':[0,1]}, 'steps':[{'range':[0,1], 'color':c}], 'bar':{'color':'white'}}))
                    fig.update_layout(height=250, margin=dict(t=50, b=0, l=10, r=10), paper_bgcolor='rgba(0,0,0,0)', font={'color':'white'})
                    return fig

                g1.plotly_chart(draw_g("Toxicity", f_tox, "orange", "tox"), use_container_width=True, key=f"t_{counter}")
                g2.plotly_chart(draw_g("Sentiment", f_sent, "blue", "sent"), use_container_width=True, key=f"s_{counter}")
                g3.plotly_chart(draw_g("Risk Score", risk, "red", "risk"), use_container_width=True, key=f"r_{counter}")

            # BOTTOM SECTION: Trend Graph spans the whole width
            st.markdown("---")
            st.subheader("Early Warning Trend Analysis")
            
            fig_trend = go.Figure(go.Scatter(x=st.session_state.history['Time'], 
                                            y=st.session_state.history['Risk'], 
                                            fill='tozeroy', mode='lines+markers', line_color='#00ccff'))
            fig_trend.update_layout(xaxis_title="Time", yaxis_title="Risk Score",
                                   xaxis=dict(tickangle=0), yaxis=dict(range=[0, 1.1]), 
                                   height=350, paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
            
            st.plotly_chart(fig_trend, use_container_width=True, key=f"trend_{counter}")

        time.sleep(2.5) 
        if not start_demo: break
else:
    st.info("👈 Use the sidebar toggle to start the infinite analysis loop.")