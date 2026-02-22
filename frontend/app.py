import os
import requests
import streamlit as st
import pandas as pd

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Car Assistant", layout="wide")
st.title("Car Repair Data Assistant (Minimal)")
st.caption("This is a plumbing check: Streamlit -> FastAPI -> response.")

with st.sidebar:
    st.subheader("Backend")
    st.write(API_BASE)

question = st.text_area(
    "Ask a question",
    value="Which province has the most repair requests?",
    height=90
)

if st.button("Ask"):
    with st.spinner("Calling backend..."):
        resp = requests.post(f"{API_BASE}/query", json={"question": question}, timeout=60)
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

    with st.expander("Show plan (debug)"):
        st.json(data.get("plan", {}))