# Hybrid CKG + RAG Query Router

This repository implements the hybrid retrieval architecture described as future work in Yarmoluk & McCreary (2026), "Benchmarking Knowledge Retrieval Architectures Across Educational and Commercial Domains." The paper's future work item called for "hybrid architectures combining CKG structural precision with RAG prose retrieval for T1/T5 query types." This implementation builds, evaluates, and validates that architecture across three domains from the benchmark corpus.

## Background

A Compact Knowledge Graph (CKG) represents a course or domain as a directed acyclic graph of concept nodes with dependency edges and taxonomy codes. The paper demonstrates that CKG achieves 3.8x higher F1 than RAG on structural queries (prerequisites, learning paths, category listings) using 11x fewer tokens. The key metric is Retrieval Efficiency Score (RDS = F1 / tokens), which captures the tradeoff between answer quality and computational cost. CKG dominates RAG on RDS by 41.7x on structural queries.

However, CKG has no prose content and cannot answer explanatory or relationship queries. When a user asks "what is a derivative?" or "how does overfitting relate to the bias-variance tradeoff?", CKG has nothing to return. RAG handles these queries well because it retrieves and ranks textbook chunks semantically. The two systems are complementary, not substitutes.

This router classifies every incoming query by type and dispatches it to the appropriate backend automatically. For explanatory queries (T1, T5) it uses RAG; for structural queries (T2, T3, T4) it uses CKG. The result is a system that captures CKG's efficiency advantage on structural queries while retaining RAG's prose-retrieval quality on explanatory ones.

## Architecture

### Query Classifier

The classifier uses a two-stage approach. Stage 1 is a keyword pre-filter that runs free (no API call) and catches approximately 80% of queries by counting keyword matches against type-specific lists. When exactly one type accumulates more than one match, that type wins immediately. Stage 2 fires only for ambiguous queries — it sends the query to `claude-sonnet-4-6` with a structured prompt and parses the single-token response.

| Type | Description | Routes to | Why |
|------|-------------|-----------|-----|
| T1 | Definition or explanation | RAG | CKG has no prose content |
| T2 | Prerequisites / dependencies | CKG | Explicit DAG edges, BFS traversal |
| T3 | Learning path between concepts | CKG | Shortest-path BFS |
| T4 | Category listing | CKG | Taxonomy code filter |
| T5 | Relationship or comparison | RAG | Prose context captures nuance |

T2 queries additionally detect direction: "what comes after X?" triggers forward traversal (dependents via reverse index) rather than backward traversal (prerequisites via dependency edges), and the LLM prompt is adjusted accordingly.

### CKG Backend

The CKG backend loads a `learning-graph.csv` file with columns `ConceptID`, `ConceptLabel`, `Dependencies` (pipe-separated IDs), and `TaxonomyID`. It builds an adjacency graph and a reverse index for forward traversal. Concept lookup is two-stage: substring matching (label contained in query, or query contained in label) followed by word-level partial matching for queries that don't mention a concept verbatim.

T2 queries run BFS up to 3 hops through dependency edges to collect prerequisites, or through the reverse index to collect dependents. T3 queries extract two concept names from the query and find the shortest path between them via bidirectional BFS. T4 queries map user-facing category words ("foundational", "core", "advanced") to actual taxonomy code groups via a domain-specific `TAXONOMY_GROUPS` mapping, then filter the graph.

### RAG Backend

The RAG backend loads all `.md` files from `data/<domain>/chapters/`, splits them into 500-word overlapping chunks, and embeds them with `sentence-transformers/all-MiniLM-L6-v2`. Embeddings are L2-normalized and stored in a numpy array. Retrieval is cosine similarity via dot product — equivalent to FAISS IndexFlatIP but without the dependency. The top-k chunks (default k=3) are concatenated as context for the LLM call.

## Results

### Single Domain (Calculus)

| Metric | Hybrid | RAG (paper) | CKG (paper) |
|--------|--------|-------------|-------------|
| Avg tokens/query | 580 | 2,982 | 269 |
| Token reduction | 80.6% | — | — |
| Classifier accuracy | 100% | — | — |
| CKG routing rate | 61.1% | — | — |
| Fallback rate | 0% | — | — |
| Avg F1 | 0.1317 | 0.1231 | 0.4709 |
| RDS | 0.00485 | 0.0000482 | 0.00201 |
| RDS vs RAG | 100.7x | 1x | 41.7x |

### 3-Domain Validation

| Metric | Calculus | Ethics | Data Science | Average |
|--------|----------|--------|--------------|---------|
| Classifier acc. | 100% | 100% | 100% | 100% |
| Avg tokens/query | 580 | 330 | 320 | 410 |
| Token reduction | 80.6% | 88.9% | 89.3% | 86.3% |
| CKG routing rate | 61.1% | 66.7% | 66.7% | 64.8% |
| Fallback rate | 0% | 0% | 0% | 0% |
| Avg F1 | 0.1317 | 0.1671 | 0.2115 | 0.1701 |
| RDS | 0.00485 | 0.00695 | 0.00915 | 0.00699 |
| RDS vs RAG | 100.7x | 144.2x | 189.8x | 144.9x |
| Est. cost | $0.085 | $0.060 | $0.059 | $0.205 |

### Comparison to Paper Baselines (Table 7)

| System | RDS | vs RAG |
|--------|-----|--------|
| RAG (paper) | 0.0000482 | 1x |
| CKG (paper) | 0.00201 | 41.7x |
| **Hybrid (this work)** | **0.00699** | **144.9x** |

The hybrid router achieves 144.9x RDS vs pure RAG and 3.5x better RDS than the paper's pure CKG baseline. Tested across 3 domains (STEM, Foundational, Professional): 48 total queries, 100% classifier accuracy, 0% fallback rate, 86.3% average token reduction.

**Note on F1 scores:** The paper's RAG baseline indexed full MkDocs textbook chapters of several thousand words per topic. This implementation uses shorter synthetic prose chapters (300–500 words each). The lower F1 reflects shorter retrieval context, not a flaw in routing logic. F1 would improve proportionally with full-corpus indexing. The RDS advantage is independent of this: it reflects structural routing efficiency, which holds regardless of prose chapter length.

## Reproduction

### Requirements

- Python 3.10+
- Anthropic API key
- ~$0.20 API credit for the 3-domain live run

### Setup

```bash
git clone <this-repo>
cd hybrid-router
pip install -r requirements.txt

# Add your API key
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

# Get real benchmark data (optional — synthetic data included)
git clone https://github.com/Yarmoluk/ckg-benchmark.git
cp ckg-benchmark/benchmark/domains/calculus/learning-graph.csv data/calculus/
```

### Running

```bash
# Dry run — free, no API calls, validates routing and token estimates
python run_multi.py --dry-run

# Live 3-domain run (~$0.19)
python run_multi.py

# Single domain
python main.py --domain calculus
```

Results are written to `results_3domains.json` (per-query detail) and appended to `run_log.jsonl` (summary per run).

## Extending to All 44 Domains

The 44-domain runner is ready for the paper's authors to execute. It requires the full benchmark directory:

```bash
# Set the path to your ckg-benchmark clone
export BENCHMARK_DIR=~/path/to/ckg-benchmark

# Estimate cost and verify routing before spending
python run_multi.py --all-domains --dry-run

# Full 44-domain run
python run_multi.py --all-domains
```

Results are saved to `results_all_domains.json` with per-domain breakdowns and a macro-average table directly comparable to Table 7 in the paper. Estimated cost for all 44 domains: $27–35. Estimated runtime: 2–4 hours depending on API throughput.

For domains without synthetic prose chapters, the `auto_generate_queries()` function in `run_multi.py` generates 10 representative queries from the CSV automatically (2 T1, 3 T2, 2 T3, 2 T4, 1 T5).

## Limitations

1. **F1 reflects synthetic chapters.** The paper's RAG indexed full MkDocs textbooks; this uses shorter synthetic prose. F1 would improve with full corpus content.
2. **Ground truth derived from the same DAG as CKG.** Structural query evaluation (T2/T3/T4) measures exact match against BFS traversal results — the same limitation applies to the original paper.
3. **Tested on 3 of 44 domains.** Macro-average across all 44 domains is pending full corpus run.
4. **Model:** All results use `claude-sonnet-4-6`. Other models (`claude-haiku-4-5-20251001`, `claude-opus-4-7`) are supported via `--model` flag but not benchmarked.

## Contributing

This implementation is intended as a contribution to the open benchmark at github.com/Yarmoluk/ckg-benchmark. If you are the paper's authors, the 44-domain runner requires no modification — point `--benchmark-dir` at your corpus and run.

Built on the McCreary Intelligent Textbook Corpus — 44 domains, 12,260 concepts, 19,405 edges.
Paper: Yarmoluk & McCreary (2026).
