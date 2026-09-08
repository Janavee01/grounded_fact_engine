import streamlit as st
import requests
import json

st.set_page_config(page_title="Fact Knowledge Layer", layout="wide")

st.title("Fact Knowledge Layer")
st.markdown("Extract and compare facts across PDF documents")

with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type=['pdf'])

    if uploaded_file and st.button("Process PDF"):
        with st.spinner("Processing..."):
            files = {'file': (uploaded_file.name, uploaded_file, 'application/pdf')}
            response = requests.post('http://localhost:8000/upload', files=files)
            if response.status_code == 200:
                st.success(f"{response.json()['message']}")
            else:
                st.error(f"Error: {response.text}")

tab1, tab2, tab3, tab4 = st.tabs(["Facts", "Compare", "Stats", "Demo Cases"])

with tab1:
    st.header("All Extracted Facts")
    response = requests.get('http://localhost:8000/facts')
    if response.status_code == 200:
        data = response.json()
        for fact in data['facts']:
            with st.expander(f"{fact['text'][:100]}..."):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Document:** {fact['source_document']}")
                    st.write(f"**Page:** {fact['source_page']}")
                    st.write(f"**Type:** {fact['fact_type']}")
                    if fact['value']:
                        st.write(f"**Value:** {fact['value']} {fact['unit'] if fact['unit'] else ''}")
                with col2:
                    st.write(f"**Confidence:** {fact['confidence']:.2%}")
                    st.write(f"**Context:** {json.dumps(fact['context'], indent=2)}")
                    st.write(f"**Snippet:** {fact['source_snippet']}")

with tab2:
    st.header("Fact Comparisons")
    response = requests.get('http://localhost:8000/compare')
    if response.status_code == 200:
        data = response.json()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Corroborations", data['summary']['corroborations_count'])
        with col2:
            st.metric("Contradictions", data['summary']['contradictions_count'])

        if data['corroborations']:
            st.subheader("Corroborations")
            for comp in data['corroborations']:
                st.info(f"{comp['explanation']}")

        if data['contradictions']:
            st.subheader("Contradictions")
            for comp in data['contradictions']:
                st.warning(f"{comp['explanation']}")

with tab3:
    st.header("Statistics")
    response = requests.get('http://localhost:8000/stats')
    if response.status_code == 200:
        data = response.json()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Facts", data['total_facts'])
        with col2:
            st.metric("Documents", data['total_documents'])
        with col3:
            st.metric("Fact Types", f"{len(data['fact_types'])}")

        st.write("**Documents:**")
        for doc in data['documents']:
            st.write(f"- {doc}")

        st.write("**Fact Types:**")
        for ftype, count in data['fact_types'].items():
            st.write(f"- {ftype}: {count}")

with tab4:
    st.header("Four Required Cases")
    st.markdown(
        "Below are the four evaluation cases. Upload at least two "
        "documents with overlapping facts and visit **Compare** to see "
        "real corroboration, contradiction, and reconciliation results."
    )

    response = requests.get("http://localhost:8000/compare")

    if response.status_code == 200:
        data = response.json()

        st.subheader("Case 1 — Corroboration")
        corrs = data.get("corroborations", [])
        if corrs:
            for c in corrs[:3]:
                st.info(f"{c['explanation']}")
        else:
            st.caption("No corroborations found yet. Upload documents with overlapping facts.")

        st.subheader("Case 2 — Contradiction")
        conts = data.get("contradictions", [])
        if conts:
            for c in conts[:3]:
                st.warning(f"{c['explanation']}")
        else:
            st.caption("No contradictions found yet.")

        st.subheader("Case 3 — Context-based Reconciliation")
        recs = data.get("reconciled", [])
        if recs:
            for c in recs[:3]:
                st.success(f"{c['explanation']}")
        else:
            st.caption("No reconciled facts found yet.")

        st.subheader("Case 4 — Extraction Failure Handling")
        st.write(
            "The grounding layer validates every LLM-claimed quote against "
            "the original PDF text using RapidFuzz fuzzy matching. Quotes "
            "that fail to ground (score < 80) are rejected before a fact "
            "is stored. This prevents hallucinated evidence from entering "
            "the knowledge base."
        )
        total = data.get("total_comparisons", 0)
        st.write(f"Comparisons evaluated: {total}")
