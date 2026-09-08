# src/comparison/comparator.py

import re
from typing import List, Dict, Any, Optional

from rapidfuzz import fuzz

from ..models.fact import FactComparison
from ..extraction.llm_client import LLMClient


COMPARISON_SYSTEM_PROMPT = """You are a precise, document-agnostic fact-comparison engine.

Compare two facts from arbitrary documents.

Do not assume any particular domain, organization, metric, terminology,
document type, unit system, or reporting convention.

Classify the relationship as exactly one of:

"corroborates"
    The facts describe the same underlying claim and are consistent.

"contradicts"
    The facts describe the same underlying claim, with sufficiently
    comparable context, but contain incompatible values or outcomes.

"reconciled"
    The facts appear different but the difference is explicitly explained
    by a difference in timeframe, scope, definition, measurement basis,
    aggregation, unit representation, rounding, or another contextual
    distinction.

"unrelated"
    The facts do not describe the same underlying claim.

Do not infer missing context.

Equivalent representations of a value should not be treated as contradictions.

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
    ) -> List[FactComparison]:

        if len(facts) < 2:
            return []

        dict_facts = [
            self._to_dict(fact)
            for fact in facts
        ]

        comparisons = []

        for index, fact1 in enumerate(dict_facts):
            for fact2 in dict_facts[index + 1:]:
                if not self._are_candidates(
                    fact1,
                    fact2,
                ):
                    continue

                comparison = self._compare_pair_with_llm(
                    fact1,
                    fact2,
                )

                if (
                    comparison
                    and comparison.relationship != "unrelated"
                ):
                    comparisons.append(comparison)

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

        document1 = f1.get(
            "source_document"
        )

        document2 = f2.get(
            "source_document"
        )

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

        entity1 = self._context_value(
            f1,
            "entity",
        )

        entity2 = self._context_value(
            f2,
            "entity",
        )

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

        # Generic token overlap.
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

            if similarity >= 0.35:
                return True

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