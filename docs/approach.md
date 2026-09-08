# Approach Details

## Why one LLM call per page instead of one call per PDF?

The whole design hinges on the grounding gate: every fact must carry a
`verbatim_quote` that fuzzy-matches the source text at score ≥ 80. That quote
is produced by the LLM *in the same prompt* that contains the source text, so
the model can copy it character-for-character.

Two tempting "optimizations" both break this:

1. **One call per entire PDF.** An 89-page report is ~200k characters. Models
   paraphrase long inputs instead of quoting exactly, so the grounding score
   collapses and facts are silently rejected. You save call-count but lose
   facts — the worse trade.
2. **Tiny chunks (750 chars).** Earlier version. Extremely grounded, but ~943
   calls for that same report. On a free-tier hosted model (20 req/min) or a
   small GPU this is hours.

**Sweet spot found empirically:** chunk at 14,000 chars ≈ one typical page.
The evidence stays in-context (grounding stays strict), and the number of calls
drops ~10x versus 750-char chunks. Text extraction (pdfplumber) is free and
local; the LLM only ever sees one page at a time.

## Why parallel workers ≠ faster on small GPUs

The dominant cost is token generation speed (tokens/sec), which is fixed by the
GPU + model size — NOT by parallelism. On a GTX 1650, `qwen3:4b` runs at ~21
tok/s; firing 8 threads just queues them onto the same GPU (and on free
hosted tiers triggers 429s). So concurrency is capped low (2-3) and the real
levers are: fewer calls (above), a faster *fits-in-VRAM* model, and the disk
cache so you never re-extract a document twice.

## Cache design

- Keyed by `sha256(pdf_bytes)[:16]` + a signature of (model, chunk budget,
  grounding thresholds, prompt version).
- Re-processing the same PDF costs zero LLM calls.
- Changing the prompt bumping the version invalidates cleanly.

## Honest failure analysis (case 4)

During development the model occasionally emitted quotes that were not
character-exact substrings of the source (e.g. reordered words, dropped
punctuation). The grounding gate catches these and *rejects the fact* — the
system does not silently return weak evidence. A real, reproducible failure:
**scanned pages with no text layer** produce zero facts for that page. Rather
than hallucinate, the pipeline logs "empty page" and continues. Adding OCR is
the documented next step.

## Numeric value handling

The LLM's `value` field is treated as untrusted. For `numeric` facts the value
is re-parsed from the grounded snippet, honoring scale words
(crore=10⁷, lakh=10⁵, million=10⁶, billion=10⁹). This is how the Delhivery
₹81,415M vs ₹8,142 crore reconciliation is kept exact rather than eyeballed.