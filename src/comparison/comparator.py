# src/comparison/comparator.py

import math
import re
from typing import List, Dict, Any, Optional, Callable

from rapidfuzz import fuzz

from ..models.fact import FactComparison
from ..extraction.llm_client import LLMClient


COMPARISON_SYSTEM_PROMPT = """You are a precise, document-agnostic fact-comparison engine.

Compare two facts from arbitrary documents.

Do not assume any particular domain, organization, metric, terminology,
document type, unit system, or reporting convention.

Work through the following five analytical steps IN ORDER. Do not skip
any of them; do not jump to a classification before completing them.

STEP 1 — EXTRACT THE CORE PROPOSITION OF EACH FACT:
  For each fact, restate its core proposition in one neutral sentence:
  WHAT is being claimed about WHOM and by HOW MUCH. Strip away surface
  wording. Example: "96 new buses in service" and "110 new buses in
  service" both concern "the number of new buses in service".

STEP 2 — EXTRACT ALL QUALIFIERS AND DIMENSIONS FROM EACH FACT:
  List, per fact and only when the fact itself provides it:
  - time / reporting period (date, fiscal year, quarter, "as of", ...)
  - scope (standalone, consolidated, phase 1, full year, subset, ...)
  - entity / population covered
  - metric / definition (what exactly is measured)
  - unit (and normalize it: crore vs lakh vs million vs billion,
    percentage vs percentage points, ...)
  - status / modality (planned vs actual, target vs achieved,
    estimate vs observed, before vs after, cumulative vs
    period-specific, revised vs original)
  - uncertainty (approximately, roughly, about, nearly, ...)
  - negation (did not occur, no, none, ...)
  If a dimension is NOT present in a fact, mark it as "not stated" —
  never invent it.

STEP 3 — ALIGN THE TWO FACTS ACROSS THOSE DIMENSIONS:
  For each dimension, determine whether the two facts agree, differ,
  are compatible after normalization, or are unknown/not stated.
  Alignment questions to answer explicitly:
  - Do they concern the same underlying proposition?
  - Do they refer to the same entity/population/metric/scope?
  - Do they refer to the same time/reporting period, or are the
    periods explicitly different?
  - Are the units compatible once normalized?
  - Is one planned/targeted/estimated and the other actual/observed?
  - Is one cumulative and the other period-specific?
  - Is one before and the other after?
  - Does negation or modality change the meaning?
  - Is any dimension unknown such that comparability cannot be
    established?

STEP 4 — DETERMINE THE RELATIONSHIP OF THE DIFFERENCES:
  After alignment, decide whether the observed differences are:
  (a) genuinely incompatible (facts explicitly assert mutually
      exclusive propositions after full alignment),
  (b) explained by contextual qualifiers (explicit information in the
      facts shows why both can be true),
  (c) independently supportive (facts substantiate the same or
      overlapping proposition),
  (d) impossible to determine from the available evidence (a needed
      dimension is not stated in either fact).

STEP 5 — ASSIGN EXACTLY ONE LABEL:
  "corroborates"
      The facts independently support substantially the same
      proposition and are consistent after alignment. Approximate
      figures that are close, or values that agree once units are
      normalized, are corroborating, not conflicting.

  "contradicts"
      After all available contextual dimensions are aligned, the facts
      explicitly assert mutually exclusive propositions. This requires
      the SAME entity, metric, scope, definition and an explicitly
      stated SAME time/reporting period in BOTH facts, with no
      contextual qualifier explaining the difference.

  "reconciled"
      The facts appear different or potentially conflicting, but
      explicit information IN the facts explains why both can be true:
      planned versus actual, target versus achieved, estimate versus
      observed, different reporting periods (both dates stated),
      cumulative progression, revised values, before/after
      measurements, different scope, approximation, different units,
      or different metric definitions.

  "insufficient_context"
      The facts differ or potentially conflict, but the available
      evidence is insufficient to establish either reconciliation or
      contradiction. Use this when a required dimension (typically the
      reporting period, or comparability of entity/metric/scope) is
      not stated in the facts. Never force CORROBORATE, RECONCILE, or
      CONTRADICT when evidence is insufficient.

  "unrelated"
      The facts concern different underlying propositions.

CRITICAL RULES:

- Different values alone do NOT imply contradiction.
- Same entity, metric, and scope alone do NOT imply contradiction.
- Planned vs actual, target vs achieved, estimate vs observed,
  different reporting periods, cumulative vs period-specific values,
  revised values, and before/after measurements MUST be evaluated
  before a contradiction can be declared.
- Never invent missing context or chronology.
- Never assume two facts refer to the same time period unless the
  evidence establishes it.
- Never infer a reporting-period difference unless BOTH facts state
  their dates/periods.
- Normalize units before comparing values.
- Distinguish percentages from percentage points.
- Respect negation, modality, uncertainty, and definitions.
- Do not infer causation merely because two facts describe related
  events or outcomes.
- If evidence is insufficient, use "insufficient_context" rather than
  forcing a label.

EXPLANATION REQUIREMENTS (critical):

- Generate the explanation ONLY AFTER assigning the classification.
- The explanation must support the assigned classification using
  evidence from BOTH facts.
- NEVER write an explanation that says the facts contradict while the
  label is "reconciled" (or vice versa).
- The explanation must not mention any internal validation or
  guardrail step. It must read as a self-contained justification
  grounded in the facts.

Use only information provided in the two facts.

Examples:

EXAMPLE 1
FACT 1 (claim): The first phase of the program was planned to cover 120 units.
FACT 2 (claim): By the end of the period, 96 units had been delivered,
rather than the 120 originally expected.
Correct label: "reconciled" - Fact 2 reports an actual outcome that
supersedes the plan described in Fact 1.

EXAMPLE 2
FACT 1 (claim): An audit counted 110 new units in service as of December 2023.
FACT 2 (claim): A review found 96 new units in service as of March 2024.
Correct label: "reconciled" - different reporting periods, so the
difference is explained by chronology.

EXAMPLE 3
FACT 1 (claim): The audit counted 110 new units in service as of December 2023.
FACT 2 (claim): A review found 96 new units in service as of December 2023.
Correct label: "contradicts" - same metric, scope, and period; mutually
incompatible; no context explains the gap.

EXAMPLE 4
FACT 1 (claim): Twenty-eight locations each received a full improvement package.
FACT 2 (claim): Some locations received the complete package while others
received only selected items.
Correct label: "corroborates" - consistent restatements with no conflict.

EXAMPLE 5
FACT 1 (claim): An audit counted 96 new units in service.
FACT 2 (claim): A review found 110 new units in service.
Correct label: "insufficient_context" - neither fact provides a reporting
period, so it cannot be confirmed whether the difference is
chronological, scope-based, or a genuine conflict.

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
have insufficient context to classify, or are unrelated.

Work through the extraction-alignment-classification steps: extract
each fact's core proposition and its qualifiers (time, scope, entity,
metric, unit, status, uncertainty, negation), align them, and only then
assign the relationship. Use only information present above.
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
        # comparison question, or an alias for the not-comparable label.
        # Normalize these to the canonical names before validating, rather
        # than silently dropping a valid result.
        relationship = {
            "corroborate": "corroborates",
            "contradict": "contradicts",
            "reconcile": "reconciled",
            "not_comparable": "insufficient_context",
            "not-comparable": "insufficient_context",
            "insufficient": "insufficient_context",
        }.get(relationship, relationship)

        allowed = {
            "corroborates",
            "contradicts",
            "reconciled",
            "insufficient_context",
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

        comparison = FactComparison(
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

        return self._apply_guardrails(
            f1,
            f2,
            comparison,
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

    # Word-level signals used to tell a stated plan/expectation apart from
    # a reported actual outcome. Generic language cues, not domain-specific.
    PLAN_CUES = re.compile(
        r"\b(plan(?:ned|s|ning)?|expected|expects|projected|proposed|"
        r"covers?|will|scheduled|target(?:ed)?|originally|planned|"
        r"anticipat(?:es|ed))\b",
        re.IGNORECASE,
    )

    ACTUAL_CUES = re.compile(
        r"\b(entered|found|recorded|reported|actual(?:ly)?|spent|"
        r"delivered|completed|received|increased|declined|received|"
        r"resulted|achieved|occurred|entered service)\b",
        re.IGNORECASE,
    )

    # "96 ... rather than the 120 expected" - an explicit plan-vs-actual
    # contrast embedded in a single claim.
    PLAN_ACTUAL_CONTRAST_RE = re.compile(
        r"\b(?:rather than|instead of|as opposed to|as against|"
        r"versus|vs\.?)\s+"
        r"(?:approximately|about|the\s+)?"
        r"(\d[\d,]*(?:\.\d+)?)",
        re.IGNORECASE,
    )

    # Generic aggregation/timeframe/scope markers that can distinguish two
    # numeric claims ("standalone" vs "consolidated", "phase 1" vs "full
    # year"). Purely linguistic; not tied to any domain.
    SCOPE_CUE_RE = re.compile(
        r"\b(standalone|consolidated|full[- ]year|annual|quarterly|"
        r"first half|second half|phase[ s0-9]*|total|overall|combined|"
        r"excluding?|including|approximately|roughly|as of|"
        r"unadjusted|adjusted|cumulative|budgeted|approved)\b",
        re.IGNORECASE,
    )

    # A cumulative measure reported at different points in time can legally
    # differ because it keeps accumulating (e.g. "cumulative units delivered
    # by May" vs "by October"). Point-in-time markers like "as of" are NOT
    # included because they only date a single snapshot.
    CUMULATIVE_CUE_RE = re.compile(
        r"\b(cumulative|year[- ]to[- ]date|ytd|running total|accrued|"
        r"to date)\b",
        re.IGNORECASE,
    )

    # A revised/restated estimate supersedes an earlier one without making
    # the two values contradictory.
    REVISED_CUE_RE = re.compile(
        r"\b(revised|revision|restated|recast|updated (?:estimate|projection|"
        r"forecast)?|upward(?:ly)? (?:revised|adjusted)|"
        r"downward(?:ly)? (?:revised|adjusted)|"
        r"superseded|initially (?:estimated|projected)|"
        r"earlier (?:estimate|projection))\b",
        re.IGNORECASE,
    )

    # Values presented as approximate/rounded: close figures should be
    # treated as consistent, not contradictory.
    APPROXIMATE_CUE_RE = re.compile(
        r"\b(approximately|approx|about|around|roughly|nearly|almost|"
        r"close to|on the order of|circa|some|several|approx\.|"
        r"~|around|estimated at)\b",
        re.IGNORECASE,
    )

    # Negation/absence markers. A claim that something "did not happen"
    # versus one that says it did are genuine contradictions only when
    # otherwise comparable; negation alone does not establish a numeric
    # contradiction when the underlying quantities are consistent.
    NEGATION_CUE_RE = re.compile(
        r"\b(not|no|never|did not|didn't|does not|doesn't|failed to|"
        r"none|lack of|absence of|without|no longer)\b",
        re.IGNORECASE,
    )

    # Causal language versus correlational language. A causal claim and a
    # merely correlated/coincidental observation are different kinds of
    # claim and should not be treated as direct contradictions of each
    # other just because the framing differs.
    CAUSAL_CUE_RE = re.compile(
        r"\b(consequently|results? in|resultant|leads? to|lead to|caused|causes|"
        r"causal|because of|due to|driven by|attributable to|"
        r"responsible for|precipitated|spurred|propelled|"
        r"brought about|account for)\b",
        re.IGNORECASE,
    )

    CORRELATION_CUE_RE = re.compile(
        r"\b(correlat(?:ed|es|ion)?|associated?|association|related?|"
        r"linked|tied to|coincid(?:ed|es)?|tracks|moves with|"
        r"accompanies|co-occur)\b",
        re.IGNORECASE,
    )

    # Percentage versus percentage-point distinction. Reporting a change
    # "from 5% to 10%" (a 5 percentage-point move) differs from reporting
    # a "5% increase" in the value.
    PERCENT_POINT_RE = re.compile(
        r"\b(percentage points?|pp|pbp)\b",
        re.IGNORECASE,
    )

    def _as_number(self, value: Any) -> Optional[float]:

        if isinstance(value, bool):
            return None

        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, str):
            cleaned = re.sub(r"[^\d.]", "", value)
            try:
                return float(cleaned)
            except ValueError:
                return None

        return None

    def _status_of(self, fact: Dict[str, Any]) -> str:

        text = self._normalize_text(
            fact.get("text")
            or fact.get("claim")
            or ""
        )

        plan = bool(self.PLAN_CUES.search(text))
        actual = bool(self.ACTUAL_CUES.search(text))

        if plan and actual:
            return "mixed"
        if plan:
            return "plan"
        if actual:
            return "actual"
        return "unknown"

    def _is_plan_actual_contrast(
        self,
        f1: Dict[str, Any],
        f2: Dict[str, Any],
    ) -> bool:

        for source in (f1, f2):
            text = self._normalize_text(
                source.get("text")
                or source.get("claim")
                or ""
            )

            match = self.PLAN_ACTUAL_CONTRAST_RE.search(text)

            if not match:
                continue

            try:
                contrast_number = float(
                    match.group(1).replace(",", "")
                )
            except ValueError:
                continue

            if self._as_number(f1.get("value")) is None:
                continue
            if self._as_number(f2.get("value")) is None:
                continue

            fact1_value = self._as_number(f1.get("value"))
            fact2_value = self._as_number(f2.get("value"))

            if fact1_value is None or fact2_value is None:
                continue

            if not math.isclose(
                contrast_number,
                fact2_value if source is f1 else fact1_value,
            ):
                continue

            # One fact carries the plan value, the other a different
            # reported result.
            if not math.isclose(fact1_value, fact2_value):
                return True

        return False

    def _scope_cues_differ(
        self,
        f1: Dict[str, Any],
        f2: Dict[str, Any],
    ) -> bool:

        text1 = self._normalize_text(
            f1.get("text")
            or f1.get("claim")
            or ""
        )
        text2 = self._normalize_text(
            f2.get("text")
            or f2.get("claim")
            or ""
        )

        cues1 = set(self.SCOPE_CUE_RE.findall(text1))
        cues2 = set(self.SCOPE_CUE_RE.findall(text2))

        return bool(
            cues1
            and cues2
            and cues1 != cues2
        )

    def _cumulative_or_revised(
        self,
        f1: Dict[str, Any],
        f2: Dict[str, Any],
    ) -> bool:

        return bool(
            self.CUMULATIVE_CUE_RE.search(
                self._normalize_text(
                    f1.get("text") or f1.get("claim")
                )
            )
            or self.CUMULATIVE_CUE_RE.search(
                self._normalize_text(
                    f2.get("text") or f2.get("claim")
                )
            )
            or self.REVISED_CUE_RE.search(
                self._normalize_text(
                    f1.get("text") or f1.get("claim")
                )
            )
            or self.REVISED_CUE_RE.search(
                self._normalize_text(
                    f2.get("text") or f2.get("claim")
                )
            )
        )

    def _has_reconciling_context(
        self,
        f1: Dict[str, Any],
        f2: Dict[str, Any],
    ) -> bool:

        time1 = self._context_value(f1, "time_period")
        time2 = self._context_value(f2, "time_period")
        time_differs = all(
            [time1, time2]
        ) and self._normalize_text(time1) != self._normalize_text(time2)

        scope1 = self._context_value(f1, "scope")
        scope2 = self._context_value(f2, "scope")
        scope_differs = all(
            [scope1, scope2]
        ) and self._normalize_text(scope1) != self._normalize_text(scope2)

        units_differ = self._units_differ(f1, f2)

        return bool(
            time_differs
            or scope_differs
            or units_differ
            or self._scope_cues_differ(f1, f2)
            or self._is_plan_actual_contrast(f1, f2)
            or self._opposite_statuses(f1, f2)
            or self._cumulative_or_revised(f1, f2)
            or self._is_approximate(f1)
            or self._is_approximate(f2)
            or self._percent_context(f1, f2)
        )

    def _is_approximate(
        self,
        fact: Dict[str, Any],
    ) -> bool:

        text = self._normalize_text(
            fact.get("text")
            or fact.get("claim")
            or ""
        )
        return bool(self.APPROXIMATE_CUE_RE.search(text))

    def _values_close(
        self,
        value1: float,
        value2: float,
        approx1: Optional[bool] = False,
        approx2: Optional[bool] = False,
    ) -> bool:

        if value1 == value2:
            return True

        # A tight rounding tolerance is accepted only when at least one
        # figure is explicitly approximate/rounded (e.g. "about 8,140"
        # vs "approximately 8,141"). Exact figures differing by more than
        # rounding noise are not silently accepted as identical.
        if approx1 or approx2:
            scale = max(abs(value1), abs(value2), 1.0)
            return bool(
                math.isclose(
                    value1,
                    value2,
                    rel_tol=0.02,
                    abs_tol=scale * 0.02,
                )
            )

        return math.isclose(value1, value2)

    def _percent_context(
        self,
        f1: Dict[str, Any],
        f2: Dict[str, Any],
    ) -> bool:

        text1 = self._normalize_text(
            f1.get("text")
            or f1.get("claim")
            or ""
        )
        text2 = self._normalize_text(
            f2.get("text")
            or f2.get("claim")
            or ""
        )

        # At least one fact expresses the value in percentage points
        # rather than as a raw percent, so the two cannot be compared
        # numerically without conversion context.
        return bool(
            self.PERCENT_POINT_RE.search(text1)
            or self.PERCENT_POINT_RE.search(text2)
        )

    def _causality_framing_differs(
        self,
        f1: Dict[str, Any],
        f2: Dict[str, Any],
    ) -> bool:

        text1 = self._normalize_text(
            f1.get("text")
            or f1.get("claim")
            or ""
        )
        text2 = self._normalize_text(
            f2.get("text")
            or f2.get("claim")
            or ""
        )

        causal1 = bool(self.CAUSAL_CUE_RE.search(text1))
        causal2 = bool(self.CAUSAL_CUE_RE.search(text2))
        corr1 = bool(self.CORRELATION_CUE_RE.search(text1))
        corr2 = bool(self.CORRELATION_CUE_RE.search(text2))

        # One fact claims causation while the other only notes a
        # correlation/outcome. These are different kinds of claim and
        # should not be forced into a direct contradiction.
        return bool(
            (causal1 and corr2)
            or (causal2 and corr1)
        )

    def _insufficient_comparability(
        self,
        f1: Dict[str, Any],
        f2: Dict[str, Any],
    ) -> bool:

        entity1 = self._context_value(f1, "entity")
        entity2 = self._context_value(f2, "entity")

        time1 = self._context_value(f1, "time_period")
        time2 = self._context_value(f2, "time_period")

        # We cannot confirm the two facts describe the same comparable
        # entity/time scope, so forcing a contradiction would be unsafe.
        entity_unknown_or_differing = bool(
            (not entity1 or not entity2)
            or self._normalize_text(entity1)
            != self._normalize_text(entity2)
        )

        time_unknown = bool(not time1 or not time2)

        both_times_unknown = bool(not time1 and not time2)

        return bool(
            entity_unknown_or_differing
            or time_unknown
        )

    def _both_times_unknown(
        self,
        f1: Dict[str, Any],
        f2: Dict[str, Any],
    ) -> bool:
        time1 = self._context_value(f1, "time_period")
        time2 = self._context_value(f2, "time_period")
        return bool(not time1 and not time2)

    def _opposite_statuses(
        self,
        f1: Dict[str, Any],
        f2: Dict[str, Any],
    ) -> bool:

        statuses = {
            self._status_of(f1),
            self._status_of(f2),
        }

        return statuses == {"plan", "actual"}

    def _units_differ(
        self,
        f1: Dict[str, Any],
        f2: Dict[str, Any],
    ) -> bool:

        unit1 = self._normalize_text(f1.get("unit"))
        unit2 = self._normalize_text(f2.get("unit"))
        return bool(
            unit1
            and unit2
            and unit1 != unit2
        )

    def _apply_guardrails(
        self,
        f1: Dict[str, Any],
        f2: Dict[str, Any],
        comparison: FactComparison,
    ) -> FactComparison:

        relationship = comparison.relationship

        value1 = self._as_number(f1.get("value"))
        value2 = self._as_number(f2.get("value"))
        values_present = bool(
            value1 is not None and value2 is not None
        )

        approx1 = self._is_approximate(f1)
        approx2 = self._is_approximate(f2)

        values_close = bool(
            values_present
            and self._values_close(
                value1,
                value2,
                approx1,
                approx2,
            )
        )
        values_conflict = bool(
            values_present
            and not values_close
        )

        has_context = self._has_reconciling_context(f1, f2)
        causality_differs = self._causality_framing_differs(f1, f2)
        times_unknown = self._both_times_unknown(f1, f2)

        # A differing measurement basis (units or percent-vs-percentage-point)
        # means even numerically equal figures are not on the same footing,
        # so they cannot simply corroborate.
        basis_differs = bool(
            self._units_differ(f1, f2)
            or self._percent_context(f1, f2)
        )

        # 1) The numeric values actually agree on the same basis (including
        # when one or both are approximate). A "contradicts" label for
        # agreeing values is a model error; the facts are consistent.
        if (
            relationship == "contradicts"
            and values_present
            and values_close
            and not basis_differs
        ):
            comparison.relationship = "corroborates"
            comparison.explanation = self._explain_for_label(
                f1,
                f2,
                "corroborates",
                "the two values are consistent (equal, or compatible "
                "allowing for rounding/approximation), so the facts agree",
            )
            return self._validate_explanation_consistency(comparison)

        # 2) A contextual distinction explains the difference, or the two
        # facts are not on the same numerical footing (units / percentage
        # points). This covers planned-versus-actual, cumulative
        # progression, revised estimates, different reporting periods,
        # different scopes/definition, approximation, units and percentage
        # points. Differing (or coincidentally equal) numeric values are NOT
        # a genuine contradiction in these cases.
        if (
            relationship == "contradicts"
            and values_present
            and (values_conflict or basis_differs)
            and has_context
        ):
            comparison.relationship = "reconciled"
            comparison.explanation = self._explain_for_label(
                f1,
                f2,
                "reconciled",
                "the differing values are explained by an explicit "
                "contextual distinction in the facts "
                "(planned-versus-actual, cumulative progression, a "
                "revised estimate, a different reporting period/scope, "
                "approximation, units, or percentage-point basis), so "
                "both facts can be true at once",
            )
            return self._validate_explanation_consistency(comparison)

        # 3) A causal claim and a correlation/outcome observation are not
        # directly contradictory even when numeric values differ slightly.
        if (
            relationship == "contradicts"
            and causality_differs
        ):
            comparison.relationship = "reconciled"
            comparison.explanation = self._explain_for_label(
                f1,
                f2,
                "reconciled",
                "one fact asserts causation while the other only notes "
                "a correlation or outcome; these are different kinds of "
                "claim and both can be true",
            )
            return self._validate_explanation_consistency(comparison)

        # 4) Both time periods are unknown: we cannot confirm the facts
        # refer to the same reporting period, so forcing a contradiction
        # would be inventing missing context.
        if (
            relationship == "contradicts"
            and values_present
            and values_conflict
            and times_unknown
            and not self._opposite_statuses(f1, f2)
        ):
            comparison.relationship = "insufficient_context"
            comparison.explanation = self._explain_for_label(
                f1,
                f2,
                "insufficient_context",
                "neither fact provides a reporting period, so it cannot "
                "be established whether the values refer to the same "
                "time scope; the difference may be chronological, "
                "scope-based, or actual",
            )
            return self._validate_explanation_consistency(comparison)

        # 5) Do not force a contradiction when the facts cannot be
        # confirmed to describe the same comparable entity/time scope.
        if (
            relationship == "contradicts"
            and values_present
            and values_conflict
            and not has_context
            and self._insufficient_comparability(f1, f2)
        ):
            comparison.relationship = "insufficient_context"
            comparison.explanation = self._explain_for_label(
                f1,
                f2,
                "insufficient_context",
                "the values differ, but the available context is not "
                "enough to confirm the facts refer to the same entity, "
                "metric, scope, and comparable time period",
            )
            return self._validate_explanation_consistency(comparison)

        # 6) The model claims a reconciliation, but neither fact supplies
        # the distinguishing context. Either the values really are
        # incompatible (contradiction) or the facts simply restate the
        # same thing consistently (corroboration).
        if relationship == "reconciled" and not has_context:
            fallback = (
                "contradicts"
                if values_conflict
                else "corroborates"
            )
            comparison.relationship = fallback
            if fallback == "contradicts":
                reason = (
                    "the values are mutually incompatible and neither "
                    "fact supplies any distinguishing context"
                )
            else:
                reason = (
                    "the values are consistent and neither fact supplies "
                    "any distinguishing context"
                )
            comparison.explanation = self._explain_for_label(
                f1,
                f2,
                fallback,
                reason,
            )
            return self._validate_explanation_consistency(comparison)

        return self._validate_explanation_consistency(comparison)

    def _clean_claim(self, fact: Dict[str, Any]) -> str:
        text = fact.get("text") or fact.get("claim") or ""
        return str(text).strip().rstrip(" .")

    def _explain_for_label(
        self,
        f1: Dict[str, Any],
        f2: Dict[str, Any],
        new_label: str,
        reason: str,
    ) -> str:

        claim1 = self._clean_claim(f1)
        claim2 = self._clean_claim(f2)
        base = reason.rstrip(" .").capitalize()

        if claim1 and claim2:
            return (
                f"Fact 1 states: {claim1}. "
                f"Fact 2 states: {claim2}. {base}."
            )
        return f"{base}."

    # ------------------------------------------------------------------
    # Final consistency check: compare the assigned label against the
    # explanation. If they disagree, REVISE the label (when the
    # explanation is clearly right) or the explanation (when the label
    # is authoritative). The explanation must always support the label.
    # ------------------------------------------------------------------

    _CORROBORATION_CUES = re.compile(
        r"\b(consistent|agrees?|agree|agreement|corroborat|"
        r"support|confirm|matching|compatible|aligned?|alike|"
        r"the same as|both (?:state|report|indicate|show))\b",
        re.IGNORECASE,
    )

    _CONTRADICTION_CUES = re.compile(
        r"\b(conflict|incompatib|mutually exclusive|"
        r"contradict|disagree|discrepan|contrary)\b",
        re.IGNORECASE,
    )

    _RECONCILIATION_CUES = re.compile(
        r"\b(reconcil|explained by|explained through|different "
        r"(?:scope|period|time|unit|definition|basis|measurement|"
        r"reporting|estimat|plan|actual|cumulative)|"
        r"planned.vs.actual|revised|restated|approximate|"
        r"rounding|progression|chronolog|can both be true|"
        r"standalone|consolidated)\b",
        re.IGNORECASE,
    )

    _INSUFFICIENT_CUES = re.compile(
        r"\b(insufficient|cannot (?:determine|confirm|establish|"
        r"verify)|not enough|unclear|unknown|missing|"
        r"lack of context|no (?:information|evidence|basis)|"
        r"unable to (?:confirm|establish|determine)|"
        r"not stated|not provided)\b",
        re.IGNORECASE,
    )

    _SUPPORTING_SENTENCES = {
        "corroborates": (
            "The two facts independently support substantially the "
            "same proposition and are consistent after alignment."
        ),
        "contradicts": (
            "The facts assert mutually incompatible claims for the same "
            "entity, metric, scope, and time period, and no contextual "
            "distinction explains the difference."
        ),
        "reconciled": (
            "The apparent difference between the facts is explained by "
            "explicit context present in the facts."
        ),
        "insufficient_context": (
            "The available evidence is insufficient to establish whether "
            "the facts reconcile or contradict."
        ),
    }

    def _label_from_explanation(
        self,
        explanation: str,
        fallback: Optional[str] = None,
    ) -> Optional[str]:

        lowered = explanation.lower()

        labels = [
            ("insufficient_context", self._INSUFFICIENT_CUES),
            ("contradicts", self._CONTRADICTION_CUES),
            ("reconciled", self._RECONCILIATION_CUES),
            ("corroborates", self._CORROBORATION_CUES),
        ]

        matched = [
            label
            for label, cue in labels
            if cue.search(lowered)
        ]

        if not matched:
            return fallback

        # If several labels match, keep the first one that also appears
        # verbatim (strongest signal), otherwise use the priority order.
        for label in matched:
            if f'"{label}"' in lowered or label.replace("_", " ") in lowered:
                return label

        return matched[0]

    def _validate_explanation_consistency(
        self,
        comparison: FactComparison,
    ) -> FactComparison:

        label = comparison.relationship

        if not comparison.explanation.strip():
            return comparison

        explanation_lower = comparison.explanation.lower()

        label_cue_map = {
            "corroborates": self._CORROBORATION_CUES,
            "contradicts": self._CONTRADICTION_CUES,
            "reconciled": self._RECONCILIATION_CUES,
            "insufficient_context": self._INSUFFICIENT_CUES,
        }

        own_cue = label_cue_map.get(label)
        own_cue_hit = bool(own_cue and own_cue.search(explanation_lower))

        reasons = []

        # REVISE THE LABEL when the explanation unambiguously supports a
        # DIFFERENT relationship and the assigned label shows none of its
        # own cues. The explanation then becomes the authoritative signal.
        if not own_cue_hit:
            explained_label = self._label_from_explanation(
                comparison.explanation
            )
            if explained_label and explained_label != label:
                comparison.relationship = explained_label
                label = explained_label
                own_cue_hit = bool(
                    label_cue_map.get(label)
                    and label_cue_map[label].search(explanation_lower)
                )

        # REVISE THE EXPLANATION when it has no support for the final
        # label: append a label-aligned sentence grounded in the facts.
        final_cue = label_cue_map.get(label)
        if final_cue and not final_cue.search(comparison.explanation.lower()):
            comparison.explanation = (
                f"{comparison.explanation.rstrip(' .')}. "
                f"{self._SUPPORTING_SENTENCES[label]}"
            )

        return comparison

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
