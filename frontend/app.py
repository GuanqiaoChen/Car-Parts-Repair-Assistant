import os
import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="LLM Data Analyst Agent", layout="wide")
st.title("LLM Data Analyst Agent")
st.caption("User Question → LLM Planner → Safe Pandas Execution → Text / Table / Chart")

with st.sidebar:
    st.subheader("Backend")
    st.write(API_BASE)
    if st.button("Health check"):
        try:
            r = requests.get(f"{API_BASE}/health", timeout=10)
            st.json(r.json())
        except Exception as e:
            st.error(str(e))

EXAMPLES = [
    "What is the distribution of repair demand types across different vehicle models and countries?",
    "Which province has the most repair requests, and which pay type is most common there?",
    "How does the average time between a vehicle's build date and its first repair demand vary by vehicle model?",
    "Show the monthly trend of total Quantity requested.",
    "Make a heatmap of DemandType by VehicleModel using row counts."
]

if "q" not in st.session_state:
    st.session_state["q"] = EXAMPLES[0]

q = st.text_area("Ask a question", key="q", height=90)

def set_example(text: str):
    st.session_state["q"] = text

cols = st.columns(len(EXAMPLES))
for i, ex in enumerate(EXAMPLES):
    cols[i].button(
        f"Example {i+1}",
        on_click=set_example,
        args=(ex,),
        key=f"ex_btn_{i}",
    )

if st.button("Ask"):
    with st.spinner("Planning and executing..."):
        resp = requests.post(
            f"{API_BASE}/query",
            json={"question": st.session_state["q"]},
            timeout=120
        )
        resp.raise_for_status()
        data = resp.json()

    st.subheader("Narrative")
    st.write(data.get("narrative", ""))

    if data.get("text"):
        st.subheader("Text")
        st.write(data["text"])

    if data.get("table"):
        st.subheader("Table")
        t = data["table"]
        df = pd.DataFrame(t["rows"], columns=t["columns"])
        st.dataframe(df, use_container_width=True)

    if data.get("chart"):
        st.subheader("Chart")
        st.plotly_chart(data["chart"], use_container_width=True)

    with st.expander("Show generated plan (debug)"):
        st.json(data.get("plan", {}))
