import requests
import json
import time

BASE_URL = "http://localhost:8000"


def upload_pdf(pdf_path):
    """Upload a PDF and return the response"""
    with open(pdf_path, 'rb') as f:
        files = {'file': (pdf_path.split('/')[-1], f, 'application/pdf')}
        response = requests.post(f"{BASE_URL}/upload", files=files)
    return response.json()


def demonstrate_cases():
    print("=" * 60)
    print("FACT KNOWLEDGE LAYER - DEMO EVALUATION")
    print("=" * 60)

    # 1. Fetch current facts
    print("\n[1] Querying Extracted Facts...")
    try:
        response = requests.get(f"{BASE_URL}/facts")
        facts_data = response.json()
        total_facts = facts_data.get('count', len(facts_data.get('facts', [])))
        print(f"--> Found {total_facts} grounded facts in knowledge base.")
    except Exception as e:
        print(f"Error fetching facts: {e}")
        return

    # 2. Run Comparison Endpoint
    print("\n[2] Querying Comparisons Across Documents...")
    try:
        response = requests.get(f"{BASE_URL}/compare")
        comp = response.json()
    except Exception as e:
        print(f"Error fetching comparisons: {e}")
        return

    # CASE 1: Corroboration
    print("\n" + "=" * 60)
    print("CASE 1: CORROBORATED FACTS")
    print("=" * 60)
    corrs = comp.get("corroborations", [])
    if corrs:
        print(f"Found {len(corrs)} corroborating relationships:")
        for idx, item in enumerate(corrs[:3], 1):
            print(f"\n  ({idx}) Fact {item['fact1_id']} <--> Fact {item['fact2_id']}")
            print(f"      Explanation: {item['explanation']}")
            print(f"      Confidence:  {item['confidence']}")
    else:
        print("No corroborating pairs found across the uploaded documents.")

    # CASE 2: Contradiction
    print("\n" + "=" * 60)
    print("CASE 2: CONTRADICTIONS")
    print("=" * 60)
    conts = comp.get("contradictions", [])
    if conts:
        print(f"Found {len(conts)} contradicting relationships:")
        for idx, item in enumerate(conts[:3], 1):
            print(f"\n  ({idx}) Fact {item['fact1_id']} <--> Fact {item['fact2_id']}")
            print(f"      Explanation: {item['explanation']}")
            print(f"      Context Notes: {item.get('context_notes')}")
    else:
        print("No direct contradictions detected.")

    # CASE 3: Reconciliation
    print("\n" + "=" * 60)
    print("CASE 3: CONTEXT-BASED RECONCILIATION")
    print("=" * 60)
    recs = comp.get("reconciled", [])
    if recs:
        print(f"Found {len(recs)} reconciled relationships:")
        for idx, item in enumerate(recs[:3], 1):
            print(f"\n  ({idx}) Fact {item['fact1_id']} <--> Fact {item['fact2_id']}")
            print(f"      Explanation: {item['explanation']}")
            print(f"      Reconciliation Context: {item.get('context_notes')}")
    else:
        print("No differing scopes or multi-period reconciliations detected in this run.")

    # CASE 4: Verification & Failure Handling (Grounding & Rejection)
    print("\n" + "=" * 60)
    print("CASE 4: HALLUCINATION & EXTRACTION FAILURE HANDLING")
    print("=" * 60)
    print("System uses RapidFuzz fuzzy grounding against the raw PDF text.")
    print("• High-confidence grounding threshold: >= 50.0")
    print("• Ungrounded hallucinated quotes or unparseable blocks are rejected and skipped.")
    print("• Generalizable: Document-agnostic prompt replaces static regex/keyword lists.")

    # Summary
    print("\n" + "=" * 60)
    print("SYSTEM SUMMARY")
    print("=" * 60)
    try:
        stats = requests.get(f"{BASE_URL}/stats").json()
        print(f"Total Documents Ingested: {stats.get('total_documents')}")
        print(f"Total Grounded Facts:     {stats.get('total_facts')}")
        print(f"Fact Types:               {json.dumps(stats.get('fact_types', {}), indent=2)}")
    except Exception as e:
        print(f"Could not load stats: {e}")

    print("\nEvaluation run complete.")


if __name__ == "__main__":
    demonstrate_cases()