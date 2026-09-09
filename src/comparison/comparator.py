# src/comparison/comparator.py

import re
from typing import List, Dict, Any, Optional, Callable

from rapidfuzz import fuzz

from ..models.fact import FactComparison
from ..extraction.llm_client import LLMClient


COMPARISON_SYSTEM_PROMPT = """You are a precise, document-agnostic fact-comparison engine.

Compare two facts from arbitrary documents.

Do not assume any particular domain, organization, metric, terminology,
document type, unit system, or reporting convention.

First determine whether the facts concern the same underlying claim.

Classify the relationship as exactly one of:

"corroborates"
    The facts describe the same underlying claim and are consistent,
    even if the wording or representation differs.

"contradicts"
    The facts describe the same underlying claim, have sufficiently
    comparable context, and contain incompatible values or outcomes.

"reconciled"
    The facts concern the same underlying claim, but an apparent
    difference is explained by an explicit contextual distinction such
    as timeframe, scope, definition, measurement basis, aggregation,
    unit representation, rounding, planned-versus-actual status, or
    another distinction supported by the provided evidence.

"unrelated"
    The facts clearly concern different underlying claims.

Do not infer missing context.

Do not treat different wording as a contradiction.

Do not treat different time periods, scopes, measurement methods,
or planned-versus-actual status as contradictions when the supplied
context explains the difference.

Use only information provided in the two facts.

Return ONLY:

{
  "relationship": "...",
  "confidence": 0.0,
  "explanation": "...",
  "context_notes": null
}
"""


class FactComparator:

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
    ):
        self.llm = llm_client or LLMClient()

    def compare_facts(
        self,
        facts: List[Any],
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[FactComparison]:

        if len(facts) < 2:
            return []

        dict_facts = [
            self._to_dict(fact)
            for fact in facts
        ]

        comparisons = []
        total_pairs = len(dict_facts) * (len(dict_facts) - 1) // 2
        completed_pairs = 0

        for index, fact1 in enumerate(dict_facts):
            for fact2 in dict_facts[index + 1:]:
                completed_pairs += 1
                if progress_callback:
                    progress_callback(
                        completed_pairs,
                        total_pairs,
                        "Filtering candidate facts",
                    )

                if not self._are_candidates(
                    fact1,
                    fact2,
                ):
                    continue

                if progress_callback:
                    progress_callback(
                        completed_pairs,
                        total_pairs,
                        "Asking the model to compare a candidate pair",
                    )

                comparison = self._compare_pair_with_llm(
                    fact1,
                    fact2,
                )

                if (
                    comparison
                    and comparison.relationship != "unrelated"
                ):
                    comparisons.append(comparison)

        if progress_callback:
            progress_callback(total_pairs, total_pairs, "Comparison complete")

        return comparisons

    def _to_dict(
        self,
        fact: Any,
    ) -> Dict[str, Any]:

        if hasattr(fact, "model_dump"):
            data = fact.model_dump()

        elif isinstance(fact, dict):
            data = dict(fact)

        elif hasattr(fact, "__dict__"):
            data = dict(fact.__dict__)

        else:
            return {}

        context = data.get("context")

        if not isinstance(context, dict):
            context = {}

        for field in (
            "entity",
            "time_period",
            "scope",
        ):
            if field not in data:
                data[field] = context.get(field)

        return data

    def _are_candidates(
        self,
        f1: Dict[str, Any],
        f2: Dict[str, Any],
    ) -> bool:

        if not f1 or not f2:
            return False

        document1 = f1.get("source_document")
        document2 = f2.get("source_document")

        if (
            document1
            and document2
            and document1 == document2
        ):
            return False

        claim1 = self._normalize_text(
            f1.get("text")
            or f1.get("claim")
            or ""
        )

        claim2 = self._normalize_text(
            f2.get("text")
            or f2.get("claim")
            or ""
        )

        if not claim1 or not claim2:
            return False

        # Keep cheap lexical candidate checks first.
        entity1 = self._context_value(f1, "entity")
        entity2 = self._context_value(f2, "entity")

        if entity1 and entity2:
            entity_similarity = fuzz.token_set_ratio(
                self._normalize_text(entity1),
                self._normalize_text(entity2),
            ) / 100.0

            if entity_similarity >= 0.80:
                return True

        claim_similarity = fuzz.token_set_ratio(
            claim1,
            claim2,
        ) / 100.0

        if claim_similarity >= 0.72:
            return True

        tokens1 = self._tokens(claim1)
        tokens2 = self._tokens(claim2)

        if tokens1 and tokens2:
            intersection = tokens1 & tokens2
            union = tokens1 | tokens2

            similarity = (
                len(intersection) / len(union)
                if union
                else 0.0
            )

            # Candidate selection is a recall gate, not the final decision.
            # A claim and a subsequent review can share only the metric and
            # its planned value (e.g. a reported outcome versus a plan), so
            # 0.35 incorrectly excludes useful cross-document pairs. The LLM
            # comparison below still rejects unrelated candidates.
            if similarity >= 0.20:
                return True

        # Semantic detection is expensive. Do not ask the model to classify
        # every possible cross-document pair; reserve it for pairs with a
        # meaningful lexical signal that did not meet the recall gate above.
        # This remains document-agnostic while keeping /compare practical on
        # a local model.
        if claim_similarity >= 0.60:
            return self._semantic_candidate_check(f1, f2)

        return False


    def _semantic_candidate_check(
        self,
        f1: Dict[str, Any],
        f2: Dict[str, Any],
    ) -> bool:

        prompt = f"""
Determine whether these two facts are plausible candidates for
comparison.

The purpose of this step is ONLY to determine whether they could
refer to the same underlying subject, metric, event, quantity,
outcome, or claim.

Important rules:

- Different wording can express the same fact.
- Synonyms and paraphrases should be considered.
- Different time periods do NOT automatically make facts unrelated.
- Different scopes do NOT automatically make facts unrelated.
- Different measurement methods do NOT automatically make facts unrelated.
- Different units or representations do NOT automatically make facts unrelated.
- Planned versus actual values may still concern the same underlying fact.
- Do not decide whether the facts corroborate, contradict, or reconcile.
- Do not reject a candidate merely because context differs.
- Reject only when the facts clearly concern different subjects,
  metrics, events, entities, or claims.

FACT 1:
Claim: {f1.get("text") or f1.get("claim")}
Fact type: {f1.get("fact_type")}
Value: {f1.get("value")}
Unit: {f1.get("unit")}
Entity: {self._context_value(f1, "entity")}
Time period: {self._context_value(f1, "time_period")}
Scope: {self._context_value(f1, "scope")}

FACT 2:
Claim: {f2.get("text") or f2.get("claim")}
Fact type: {f2.get("fact_type")}
Value: {f2.get("value")}
Unit: {f2.get("unit")}
Entity: {self._context_value(f2, "entity")}
Time period: {self._context_value(f2, "time_period")}
Scope: {self._context_value(f2, "scope")}

Return ONLY valid JSON:

{{
  "candidate": true
}}
"""

        try:
            result = self.llm.generate_json(
                system_prompt=(
                    "You are a broad semantic candidate detector for "
                    "a document-agnostic fact knowledge system. "
                    "Prefer recall at this stage. Context differences "
                    "must not be treated as a reason to reject a candidate."
                ),
                user_prompt=prompt,
                temperature=0.0,
            )

            if not isinstance(result, dict):
                return False

            candidate = result.get("candidate")

            if isinstance(candidate, bool):
                return candidate

            if isinstance(candidate, str):
                return candidate.strip().lower() == "true"

            return False

        except Exception as exc:
            print(
                f"[comparator] Semantic candidate check failed: {exc}"
            )
            return False

    def _compare_pair_with_llm(
        self,
        f1: Dict[str, Any],
        f2: Dict[str, Any],
    ) -> Optional[FactComparison]:

        user_prompt = f"""
FACT 1

Claim:
{f1.get("text") or f1.get("claim")}

Fact type:
{f1.get("fact_type")}

Value:
{f1.get("value")}

Unit:
{f1.get("unit")}

Entity:
{self._context_value(f1, "entity")}

Time period:
{self._context_value(f1, "time_period")}

Scope:
{self._context_value(f1, "scope")}

Source:
{f1.get("source_document")}

Page:
{f1.get("source_page")}


FACT 2

Claim:
{f2.get("text") or f2.get("claim")}

Fact type:
{f2.get("fact_type")}

Value:
{f2.get("value")}

Unit:
{f2.get("unit")}

Entity:
{self._context_value(f2, "entity")}

Time period:
{self._context_value(f2, "time_period")}

Scope:
{self._context_value(f2, "scope")}

Source:
{f2.get("source_document")}

Page:
{f2.get("source_page")}


Determine whether these facts corroborate, contradict, reconcile,
or are unrelated.

Use only information present above.
"""

        try:
            result = self.llm.generate_json(
                system_prompt=COMPARISON_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.0,
            )

        except Exception as exc:
            print(
                f"[comparator] Skipping pair due to error: {exc}"
            )
            return None

        if not isinstance(result, dict):
            return None

        relationship = str(
            result.get(
                "relationship",
                "unrelated",
            )
        ).strip().lower()

        # Models occasionally return the imperative form used in the
        # comparison question. Normalize it to the API's documented labels
        # before validating, rather than silently dropping a valid result.
        relationship = {
            "corroborate": "corroborates",
            "contradict": "contradicts",
            "reconcile": "reconciled",
        }.get(relationship, relationship)

        allowed = {
            "corroborates",
            "contradicts",
            "reconciled",
            "unrelated",
        }

        if relationship not in allowed:
            relationship = "unrelated"

        try:
            confidence = float(
                result.get(
                    "confidence",
                    0.0,
                )
            )
        except (TypeError, ValueError):
            confidence = 0.0

        confidence = max(
            0.0,
            min(1.0, confidence),
        )

        fact1_id = str(
            f1.get("id")
            or f1.get("fact_id")
            or "f1"
        )

        fact2_id = str(
            f2.get("id")
            or f2.get("fact_id")
            or "f2"
        )

        return FactComparison(
            fact1_id=fact1_id,
            fact2_id=fact2_id,
            relationship=relationship,
            confidence=confidence,
            explanation=str(
                result.get(
                    "explanation",
                    "",
                )
            ),
            context_notes=result.get(
                "context_notes"
            ),
        )

    def _context_value(
        self,
        fact: Dict[str, Any],
        field: str,
    ) -> Any:

        direct = fact.get(field)

        if direct not in (
            None,
            "",
            [],
            {},
        ):
            return direct

        context = fact.get("context")

        if isinstance(context, dict):
            return context.get(field)

        return None

    def _normalize_text(
        self,
        value: Any,
    ) -> str:

        if value is None:
            return ""

        text = str(value).lower()

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    def _tokens(
        self,
        text: str,
    ) -> set:

        return set(
            re.findall(
                r"\b\w+\b",
                text,
            )
        )
