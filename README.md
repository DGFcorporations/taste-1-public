# taste-1

TASTE variant 1 is a reference implementation of an AI personalization engine. It solves the problem of tailoring LLM outputs to specific personal or organizational standards without massive prompt bloat. It does this by using atomic micro-layers and cosine retrieval to inject only the most relevant instructions (and historical rationale) into the context window for a given task, then scores the candidates based on deterministic rules to ensure compliance.

## Stack
- **Language:** Python 3.11+
- **API Framework:** FastAPI
- **Embeddings:** FastEmbed (all-MiniLM-L6-v2)
- **Storage:** SQLite WAL (with packed f32 embeddings)
- **Integration:** Model Context Protocol (MCP)

## How to Run

Install in editable mode with the serving extras:
```bash
pip install -e ".[serve]"
```

Run the FastAPI server using Granian (or your preferred ASGI server):
```bash
TASTE_APP=taste.api:create_app granian --interface asgi taste.api:app
```

Alternatively, use the CLI for direct interaction:
```bash
taste serve
```

## Key Features
- **Cosine Top-N Retrieval:** Simplest honest semantic retrieval (n=12, sim≥0.40) to find the right standards.
- **Atomic Micro-Layers:** Proven schema shape for granular, specific directives.
- **Deterministic Rules Scorer:** Enforces compliance through quoted-string violations and cosine-to-centroid scoring at zero token cost.
- **Adaptive Interviewing:** The engine asks only what the existing corpus cannot resolve, getting the freshest signal per task.
- **Direct Extraction Memory:** The fastest learning loop, mapping decisions directly to provisional why-layers.

## Corpus Note

The public mechanics of TASTE variant 1 — engine code and docs only.

**What's never here:** the owner's corpus (standards, decision records, rationale, persona-derived layers). Those live in the private repo and are the system's learned knowledge — publishing them would publish the person.

Each trial cycle: corpus pruned → keepers stay private → mechanics worth sharing land here via the publish decision.
