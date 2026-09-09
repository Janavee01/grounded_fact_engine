# Comparison — Cross-Document Fact Relationships

Deep-dive on the comparison engine: how facts from different documents are
paired, how the system decides whether they corroborate, contradict, or
reconciled, and how the LLM explains each relationship.

> The comparator is **document-agnostic** - it makes no assumptions about the
> content, domain, filenames, or schema of the documents it compares.

## 1. Purpose

Given facts extracted (and grounded) from multiple PDFs, the comparator:

1. **Pairs** candidate facts across **different** documents.
2. **Classifies** each pair as `corroborates`, `contradicts`, `reconciled`, or
   `unrelated`.
3. **Explains** the relationship with an LLM-written explanation and a confidence
   score.

## 2. Flow

```
       all facts (across documents)
                 │
                 ▼
   ┌───────────────────────────────────────────────┐
   │  PRE-FILTER (candidate selection)            │
   │  • skip pairs from the SAME document         │
   │  • gate (any one must hold):                │
   │      entity similarity          ≥ 80%      │
   │      claim similarity           ≥ 72%      │
   │      Jaccard token overlap      ≥ 20%      │
   └───────────────────────────────────────────────┘
                 │
                 ▼
   ┌───────────────────────────────────────────────┐
   │  LLM CLASSIFICATION (document-agnostic)      │
   │                                             │
   │   corroborates  ┌─ same fact, different     │
   │   contradicts   │   wording / values        │
   │   reconciled    ├─ appeared to differ,      │
   │   unrelated     │   resolved via units      │
   │                 │   / scope / definition    │
   │                 └─ no relationship          │
   │                                             │
   │   outputs: explanation + confidence          │
   │   (confidence clamped to [0.0, 1.0])         │
   └───────────────────────────────────────────────┘
                 │
                 ▼
   unrelated pairs filtered out of returned results
   (only corroborate / contradict / reconciled returned)
```

## 3. Candidate Selection (Pre-Filter)

Comparisons are expensive — every pair-of-candidates across all documents is
considered. To avoid noise and keep results meaningful, pairs are gated before
they reach the LLM:

- **Same-document pairs are rejected early** — comparing a fact to another fact
  from the same document (or itself) is trivial and uninteresting.
- A pair is a candidate if **any** of these hold:
  - entity similarity **≥ 80%**, or
  - claim similarity **≥ 72%**, or
  - Jaccard token overlap **≥ 20%**.
- Pairs that score **≥ 60%** on claim similarity but miss the gates above go
  through a cheap **semantic candidate check**: one extra LLM call that only
  decides whether the two facts *could* refer to the same underlying subject —
  it never classifies the relationship.
- The matcher pulls `entity`, `time_period`, and `scope` from each fact's
  context dict so matching uses the real attributes rather than raw text.

## 4. The Four Outcomes

| Outcome | Meaning | Example |
|---------|---------|---------|
| `corroborates` | Same fact stated differently (values consistent within rounding / vintage) | India FY25 GDP growth: ES 6.4% vs IMF ~6.5% |
| `contradicts` | Same metric/entity/period with incompatible values | Two sources report incompatible GDP growth for the same period |
| `reconciled` | Apparent contradiction explained (e.g. unit or definition difference) | FY24 revenue ₹81,415M vs ₹8,142 crore → converted, same figure |
| `unrelated` | No meaningful relationship | Filtered out of returned results |

### Reconciliation example (uses numeric re-derivation)

The comparator recognizes that `₹81,415 million` and `₹8,142 crore` are the same
figure after unit conversion (crore → million), classifying the pair as
`reconciled` with an explanation of the unit convention.

## 5. LLM Classification Prompt

The classification prompt is **document-agnostic**: it contains no domain- or
industry-specific assumptions. It receives the full context of both facts
(entity, claim, time period, scope, unit, value, verbatim quote) and instructs
the model to:

- pick exactly one of the four outcomes,
- write a concise human-readable `explanation`,
- output a `confidence` in `[0.0, 1.0]`.

Confidence is clamped to `[0.0, 1.0]`; unknown/absent values default to `0.0`.

## 6. Output Contract

A relationship result carries:

```
{
  "fact_a":        { ... grounded fact ... },
  "fact_b":        { ... grounded fact ... },
  "relationship":  "corroborates" | "contradicts" | "reconciled",
  "explanation":   "...",
  "confidence":    0.0 .. 1.0,
  "source_snippet":  evidence quoted from each document
}
```

Only non-`unrelated` pairs are exposed through the API/UI.