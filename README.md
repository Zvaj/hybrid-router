# Hybrid CKG + RAG Query Router

Implementation of the hybrid retrieval architecture listed as
future work in Yarmoluk & McCreary (2026) — "Benchmarking
Knowledge Retrieval Architectures Across Educational and
Commercial Domains."

## Why this exists

The paper proves that Compact Knowledge Graphs (CKG) achieve
3.8x higher F1 than RAG on structural queries using 11x fewer
tokens. However CKG has no prose content and cannot answer
explanatory queries. RAG handles those well but is expensive
for structural queries.

No system existed that routes between the two based on query
type. This router classifies every incoming query as structural
(T2/T3/T4 → CKG) or explanatory (T1/T5 → RAG) and sends it
to the appropriate backend automatically.

## How it works

| Type | Description | Routes to | Paper CKG F1 | Paper RAG F1 |
|------|-------------|-----------|--------------|--------------|
| T1 | Definition or explanation | RAG | 0.207 | 0.094 |
| T2 | Prerequisites/dependencies | CKG | 0.634 | 0.078 |
| T3 | Learning path between concepts | CKG | 0.660 | 0.201 |
| T4 | List all concepts of a category | CKG | 0.964 | 0.286 |
| T5 | Relationship between concepts | RAG | 0.323 | 0.115 |

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Add your Anthropic API key
echo "ANTHROPIC_API_KEY=your_key_here" > .env

# Dry run first — free, no API calls
python main.py --dry-run

# Real run — approx $0.12
python main.py --domain calculus
```

## Getting the real benchmark data

```bash
git clone https://github.com/Yarmoluk/ckg-benchmark.git
mkdir -p data/calculus/chapters
cp ckg-benchmark/benchmark/domains/calculus/learning-graph.csv \
   data/calculus/
```

## Results — single domain (calculus)

| Metric | Hybrid | RAG (paper) | CKG (paper) |
|--------|--------|-------------|-------------|
| Avg tokens/query | 486 | 2,982 | 269 |
| Token reduction | 83.7% | — | — |
| Classifier accuracy | 100% | — | — |
| Avg F1 | 0.0875 | 0.1231 | 0.4709 |
| RDS | 0.00232 | 0.0000482 | 0.00201 |
| RDS vs RAG | 48.1x | 1x | 41.7x |

## Results — 3 domains (live run)

| Metric | Calculus | Ethics | Data Science | Average |
|--------|----------|--------|--------------|---------|
| Classifier acc. | 100% | 100% | 100% | 100% |
| Avg tokens/query | 486 | 188 | 203 | 292 |
| Token reduction | 83.7% | 93.7% | 93.2% | 90.2% |
| Avg F1 | 0.0875 | 0.1698 | 0.2113 | 0.1562 |
| RDS | 0.00232 | 0.00723 | 0.00898 | 0.00618 |
| RDS vs RAG | 48.1x | 150.1x | 186.4x | 128.2x |
| Cost | $0.080 | $0.053 | $0.054 | $0.19 total |

**Paper baselines (Table 7, Yarmoluk & McCreary 2026):**
RAG RDS: 0.0000482 (1x) · CKG RDS: 0.00201 (41.7x) · **Hybrid RDS: 0.00618 (128.2x)**

Tested across 3 domains (STEM, Foundational, Professional).
48 total queries — 100% classifier accuracy.
90.2% average token reduction vs RAG baseline.
128.2x average RDS improvement vs pure RAG.
3.1x better RDS than paper's pure CKG baseline.

## Cost

| Run | Actual cost |
|-----|-------------|
| Single domain (18 queries) | $0.080 |
| Three domains (48 queries) | $0.19 |
| Full 44 domains (~7,758 queries) | ~$27-35 |

## Scaling to full corpus

The router accepts a --domain flag for any domain in the
benchmark corpus:

```bash
python main.py --domain algebra
python main.py --domain biology
```

To run all 44 domains the authors can execute:

```bash
for domain in ckg-benchmark/benchmark/domains/*/; do
  name=$(basename $domain)
  python main.py --domain $name
done
```

This produces per-domain results comparable to Table 7
in Yarmoluk & McCreary (2026).

## Contributing

This implementation is intended as a contribution to the
open benchmark at github.com/Yarmoluk/ckg-benchmark.

Built on the McCreary Intelligent Textbook Corpus —
44 domains, 12,260 concepts, 19,405 edges.
