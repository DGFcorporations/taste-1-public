# VARIANT: taste-1 — the control

**Thesis under test:** the spec as designed works as designed.
This is the reference implementation every other variant is measured
against — the baseline that makes the cohort legible.

## Permutation

| Axis | Choice | Why |
|---|---|---|
| A schema | A1 atomic micro-layers | proven shape (taste_engine prior art) |
| B retrieval | B1 cosine top-N (n=12, sim≥0.40) | simplest honest semantic retrieval |
| C embedding | C1 all-MiniLM-L6-v2 | parity baseline, ~free, local ONNX |
| D scorer | D1 rules (quoted-string violations + cosine-to-centroid) | deterministic, zero cost |
| E interview | E1 adaptive (asks only what corpus can't resolve) | freshest signal per task |
| F hone | F1 BoN-3 (3 candidates, score picks) | matches the triage instinct |
| G storage | G1 SQLite WAL, packed f32 embeddings | single file, fleet pattern |
| H memory | H1 direct extraction (decision → provisional why-layer) | fastest learning loop |
| I injection | I1 directive blocks (never/do/why + exemplars) | proven prompt format |
| J pipeline | J1 fixed segments per task type | rigid mechanics law |

## What success looks like

Audit-passing, owner-recognized output from a flash model + packet alone.
If the control wins weeks 1–3, the fancy permutations have to justify their
complexity.
