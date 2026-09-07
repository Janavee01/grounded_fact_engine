import streamlit as st
import requests
import json
import os

st.set_page_config(page_title="Fact Knowledge Layer", layout="wide")

st.title("Fact Knowledge Layer")
st.markdown("Extract and compare facts across PDF documents")

# Sidebar
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

# Main content
tab1, tab2, tab3 = st.tabs(["Facts", "Compare", "Stats"])

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
    st.header("🔍 Fact Comparisons")
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

tab4 = st.tabs(["Demo Cases"])[0]  

with tab4:
    st.header("Four Required Cases Demo")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Case 1: Corroboration")
        st.info("Facts that agree across documents")
        st.code("""
        Document 1: "Revenue was $5.2M in 2025"
        Document 2: "Revenue reached $5.3M in 2025"
        -> CORROBORATES: Both report similar revenue ~$5.2M
        """)

        st.subheader("Case 2: Contradiction")
        st.warning("Facts that conflict")
        st.code("""
        Document 1: "The CEO is John Smith"
        Document 2: "The CEO is Jane Doe"
        -> CONTRADICTS: Different CEOs reported
        """)

    with col2:
        st.subheader("Case 3: Context Resolution")
        st.success("Apparent contradiction explained by context")
        st.code("""
        Document 1: "Revenue: $5.2M (Q1 2025)"
        Document 2: "Revenue: $21M (FY 2025)"
        -> RESOLVED: Q1 vs Annual - not contradictory
        """)

        st.subheader("Case 4: Extraction Failure")
        st.warning("Known limitations and improvements")
        st.code("""
        Current Failure: Complex relationships
        Example: "The system was built by X in 2020
                  and upgraded by Y in 2022"
        -> Improvement: Add LLM for relationship extraction
        """)

    # Real data from your PDF
    st.subheader("Your SRS Document Results")

    response = requests.get("http://localhost:8000/facts")

    if response.status_code == 200:
        data = response.json()

        st.write(f"Found {data['count']} facts in your SRS document")

        # Show sample facts
        st.write("Sample Requirements Extracted:")

        for fact in data["facts"][:5]:
            if fact["fact_type"] == "semantic":
                st.write(f"- {fact['text'][:150]}...")