import os
import requests
import streamlit as st
import pandas as pd

# Base URL for the FastAPI backend that owns planning and execution.
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

# Basic page chrome that advertises the NL -> LLM -> Pandas -> UI pipeline.
st.set_page_config(page_title="Data Analyst Agent", layout="wide")
st.title("Data Analyst Agent")
st.caption("LLM multi-step planning -> safe execution -> text/tables/charts output")

with st.sidebar:
    st.subheader("Backend")
    st.write(API_BASE)
    # Small health probe so users can debug connectivity without leaving the UI.
    if st.button("Health check"):
        try:
            r = requests.get(f"{API_BASE}/health", timeout=10)
            st.json(r.json())
        except Exception as e:
            st.error(str(e))

# Example prompts that reflect the kinds of questions the backend is optimised
# for; these double as documentation for non-technical users.
EXAMPLES = [
    "What is the distribution of repair demand types across different vehicle models and countries?",
    "Which province has the most repair requests, and which pay type is most common there?",
    "How does the average time between a vehicle's build date and its first repair demand vary by vehicle model?",
    "Show the monthly trend of total Quantity requested.",
    "Make a heatmap of DemandType by VehicleModel using row counts.",
    "Heatmap of VIN by SSPart using row counts.",
    "Compute correlation between Quantity and VehicleAgeDays for CountryCode == ZZZ.",
]

if "q" not in st.session_state:
    st.session_state["q"] = EXAMPLES[0]

# This is the only place where a free-form natural language query is collected;
# everything downstream treats it as data and sends it to the backend as JSON.
q = st.text_area("Ask a question", key="q", height=90)


def set_example(text: str):
    """Helper used by example buttons to populate the query text area."""
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
    # The Streamlit layer is intentionally thin: it serialises the question into
    # the 'QueryRequest' contract and lets the backend decide how to answer it.
    with st.spinner("Planning and executing..."):
        resp = requests.post(f"{API_BASE}/query", json={"question": q}, timeout=180)
        resp.raise_for_status()
        data = resp.json()

    st.subheader("Final narrative")
    st.write(data.get("final_narrative", ""))

    suggestions = data.get("suggestions", [])
    if suggestions:
        # Suggestions come straight from the backend and are rendered as a
        # small block of guidance on how to refine or extend the question.
        st.info("Suggestions:\n- " + "\n- ".join(suggestions))

    items = data.get("items", [])
    for idx, item in enumerate(items, start=1):
        st.markdown(f"## Step {idx}")

        # The backend provides a structured 'status' field so the UI can render
        # preview/error states without relying on special symbols in strings.
        status = item.get("status", "ok")
        notices = item.get("notices", []) or []

        if status == "error":
            # Errors are shown prominently; the narrative still provides context.
            if notices:
                st.error("\n".join(notices))
        elif status == "preview":
            # Preview indicates the backend applied system-level truncation.
            if notices:
                st.warning("\n".join(notices))

        st.write(item.get("narrative", ""))
        st.caption(item.get("explanation", ""))

        # The output type is chosen server-side; the frontend simply inspects
        # which fields are present and renders them in a natural order.
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