import re
from typing import List, Dict, Any, Optional
from rapidfuzz import fuzz

from ..models.fact import FactComparison
from ..extraction.llm_client import LLMClient

COMPARISON_SYSTEM_PROMPT = """You are an objective, precise fact-comparison engine.
Compare two facts extracted from documents and determine their relationship.

Carefully compare their core VALUES, ENTITIES, UNITS, and TIMEFRAMES/SCOPES:
- "contradicts": Both facts describe the same entity and metric under the same timeframe/scope, but state DIFFERENT numbers, values, or outcomes.
- "corroborates": Both facts describe the same entity/metric and agree on the same value and unit.
- "reconciled": The values or claims differ, but the discrepancy is logically explained by differing time periods, scopes, currencies, or conditions.
- "unrelated": The facts discuss different subjects or independent metrics.

Check the exact value fields carefully. If Value 1 is 20 and Value 2 is 25 for the same entity and scope, they CONTRADICT each other.

Return ONLY a JSON object in this format:
{
  "relationship": "corroborates" | "contradicts" | "reconciled" | "unrelated",
  "confidence": 0.0 to 1.0,
  "explanation": "A concise 1-2 sentence explanation pointing out the exact values and why they agree, conflict, or reconcile.",
  "context_notes": "Key difference if reconciled, else null"
}"""


class FactComparator:
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

    def compare_facts(self, facts: List[Any]) -> List[FactComparison]:
        """Compare extracted facts and identify relationships using candidate pruning + LLM reasoning."""
        comparisons: List[FactComparison] = []
        n = len(facts)
        if n < 2:
            return comparisons

        # Convert facts to dict if they are Pydantic models or objects
        dict_facts = [
            f.model_dump() if hasattr(f, "model_dump")
            else f.__dict__ if hasattr(f, "__dict__")
            else f
            for f in facts
        ]

        # 1. Candidate selection using heuristics to avoid O(n^2) LLM calls
        candidate_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                f1 = dict_facts[i]
                f2 = dict_facts[j]
                if self._are_candidates(f1, f2):
                    candidate_pairs.append((f1, f2))

        # 2. LLM comparison for shortlisted candidates
        for f1, f2 in candidate_pairs:
            comp = self._compare_pair_with_llm(f1, f2)
            if comp and comp.relationship != "unrelated":
                comparisons.append(comp)

        return comparisons

    def _are_candidates(self, f1: Dict[str, Any], f2: Dict[str, Any]) -> bool:
        """Prune pairs early without hardcoded keywords."""
        # 1. Matching or overlapping entities
        e1 = str(f1.get("entity") or "").strip().lower()
        e2 = str(f2.get("entity") or "").strip().lower()
        if e1 and e2 and (e1 in e2 or e2 in e1 or fuzz.ratio(e1, e2) > 75):
            return True

        # 2. High lexical similarity in claim text
        c1 = str(f1.get("claim") or f1.get("text") or "").lower()
        c2 = str(f2.get("claim") or f2.get("text") or "").lower()
        if fuzz.token_set_ratio(c1, c2) > 65:
            return True

        # 3. Share at least two meaningful content words (>3 chars)
        words1 = set(re.findall(r"\b\w{4,}\b", c1))
        words2 = set(re.findall(r"\b\w{4,}\b", c2))
        shared = words1.intersection(words2)
        if len(shared) >= 2:
            return True

        return False

    def _compare_pair_with_llm(self, f1: Dict[str, Any], f2: Dict[str, Any]) -> Optional[FactComparison]:
        """Hybrid comparison: code inspects value equality/inequality; LLM reasons about scope and explains."""
        val1 = str(f1.get("value") or "").strip()
        val2 = str(f2.get("value") or "").strip()
        
        # Check if values are numeric
        def try_num(v):
            clean = re.sub(r"[^\d.-]", "", v)
            try:
                return float(clean)
            except ValueError:
                return None

        num1, num2 = try_num(val1), try_num(val2)
        values_differ = (num1 != num2) if (num1 is not None and num2 is not None) else (val1.lower() != val2.lower())

        user_prompt = f"""Compare these two extracted facts:

Fact 1:
- Claim: {f1.get('claim') or f1.get('text')}
- Entity: {f1.get('entity')}
- Value: {val1}
- Unit: {f1.get('unit')}
- Time Period: {f1.get('time_period')}
- Scope: {f1.get('scope')}

Fact 2:
- Claim: {f2.get('claim') or f2.get('text')}
- Entity: {f2.get('entity')}
- Value: {val2}
- Unit: {f2.get('unit')}
- Time Period: {f2.get('time_period')}
- Scope: {f2.get('scope')}

ANALYSIS HINT:
The values are {'DIFFERENT' if values_differ else 'IDENTICAL'} ({val1} vs {val2}).
- If values are DIFFERENT and describe the same scope/timeframe, classify as 'contradicts'.
- If values are DIFFERENT but explained by differing scope, timeframe, or definitions, classify as 'reconciled'.
- If values are IDENTICAL and describe the same entity/metric, classify as 'corroborates'.
- If they describe unrelated subjects, classify as 'unrelated'.
"""

        try:
            res = self.llm.generate_json(
                system_prompt=COMPARISON_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.0
            )
        except Exception as exc:
            print(f"[comparator] Skipping pair due to error: {exc}")
            return None

        if not isinstance(res, dict):
            return None

        rel = res.get("relationship", "unrelated").lower()
        if rel not in {"corroborates", "contradicts", "reconciled", "unrelated"}:
            rel = "unrelated"

        # Deterministic sanity guard: if values strictly differ on same scope, prevent false corroboration
        if values_differ and rel == "corroborates":
            rel = "contradicts" if not (f1.get("scope") or f2.get("scope") or f1.get("time_period") != f2.get("time_period")) else "reconciled"

        f1_id = str(f1.get("id") or f1.get("fact_id") or "f1")
        f2_id = str(f2.get("id") or f2.get("fact_id") or "f2")

        return FactComparison(
            fact1_id=f1_id,
            fact2_id=f2_id,
            relationship=rel,
            confidence=float(res.get("confidence", 0.85)),
            explanation=str(res.get("explanation", "")),
            context_notes=res.get("context_notes")
        )