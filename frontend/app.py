import os
import requests
import streamlit as st
import pandas as pd

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Data Analyst Agent", layout="wide")
st.title("Data Analyst Agent")
st.caption("LLM multi-step planning → safe execution → text/tables/charts output")

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
    "Make a heatmap of DemandType by VehicleModel using row counts.",
    "Is vehicle age correlated with Quantity?",
    "Top 10 parts by Quantity and their share of total."
]

if "q" not in st.session_state:
    st.session_state["q"] = EXAMPLES[0]

q = st.text_area("Ask a question", key="q", height=90)

def set_example(text: str):
    st.session_state["q"] = text

btn_cols = st.columns(4)
for i in range(min(4, len(EXAMPLES))):
    btn_cols[i].button(
        f"Example {i+1}",
        on_click=set_example,
        args=(EXAMPLES[i],),
        key=f"ex_btn_{i}",
    )

more_cols = st.columns(3)
for j in range(3):
    idx = 4 + j
    if idx < len(EXAMPLES):
        more_cols[j].button(
            f"Example {idx+1}",
            on_click=set_example,
            args=(EXAMPLES[idx],),
            key=f"ex_btn_{idx}",
        )

if st.button("Ask"):
    with st.spinner("Planning and executing..."):
        resp = requests.post(f"{API_BASE}/query", json={"question": q}, timeout=180)
        resp.raise_for_status()
        data = resp.json()

    st.subheader("Final narrative")
    st.write(data.get("final_narrative", ""))

    suggestions = data.get("suggestions", [])
    if suggestions:
        st.info("Suggestions:\n- " + "\n- ".join(suggestions))

    items = data.get("items", [])
    for idx, item in enumerate(items, start=1):
        st.markdown(f"## Step {idx}")
        st.write(item.get("narrative", ""))
        st.caption(item.get("explanation", ""))

        if item.get("text"):
            st.write(item["text"])

        if item.get("table"):
            t = item["table"]
            df = pd.DataFrame(t["rows"], columns=t["columns"])
            st.dataframe(df, use_container_width=True)

        if item.get("chart"):
            st.plotly_chart(item["chart"], use_container_width=True)

        with st.expander("Show generated plan (debug)"):
            st.json(item.get("plan", {}))
