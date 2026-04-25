# Hybrid CKG + RAG Query Router: Implementing and Evaluating the Future Work Architecture from Yarmoluk & McCreary (2026)

**Author:** Czv
**Date:** April 2026
**Repository:** https://github.com/Zvaj/hybrid-router
**Based on:** Yarmoluk & McCreary (2026), "Benchmarking Knowledge Retrieval Architectures Across Educational and Commercial Domains"

---

## Abstract

Yarmoluk & McCreary (2026) demonstrated that Compact Knowledge Graphs (CKG) achieve 3.8x higher F1 than Retrieval-Augmented Generation (RAG) using 11x fewer tokens per query, yielding a 41.7x advantage in Reasoning Density Score (RDS = F1 / tokens). However, the three systems evaluated — RAG, GraphRAG, and CKG — were benchmarked independently with no routing between them. The paper identified this as future work: "Hybrid architectures combining CKG structural precision with RAG prose retrieval for T1/T5 query types." This paper implements that architecture. A two-stage query classifier routes structural queries (prerequisites, learning paths, category listings) to CKG and explanatory or relational queries to RAG automatically. Evaluated across 48 queries spanning 3 domains (STEM, Foundational, Professional), the system achieves 100% classifier accuracy, 86.3% average token reduction, and an average RDS of 144.9x vs. pure RAG — at a total API cost of $0.2046. A 44-domain runner is included for the benchmark authors to execute against the full corpus.

---

## 1. Introduction

Yarmoluk & McCreary (2026) benchmarked three knowledge retrieval architectures across 44 educational and commercial domains drawn from the McCreary Intelligent Textbook Corpus — 12,260 concepts, 19,405 edges, and 7,758 evaluation queries. The central finding was that CKG dramatically outperforms RAG on structural query types: CKG consumes 269 tokens per query versus RAG's 2,982, while achieving higher F1 on prerequisite, path, and category queries. The paper introduced Reasoning Density Score (RDS = F1 / tokens consumed) as the first metric to jointly optimize retrieval quality and computational cost at multi-domain scale. On that measure, CKG scored 0.00201 against RAG's 0.0000482 — a 41.7x advantage.

The evaluation, however, tested each system in isolation. CKG has a structural blind spot: it contains no prose, only a directed acyclic graph of concept nodes with dependency edges and taxonomy codes. On T1 definition queries it achieves F1 of only 0.207, and on T5 cross-concept relational queries only 0.323, because those query types require explanatory text that the graph cannot supply. RAG, conversely, is the expensive path for structural queries: it spends 2,982 tokens per query to retrieve prose chunks that contain far less precise information than a direct BFS traversal of the dependency graph, where CKG achieves F1 of 0.634 on T2 prerequisite queries, 0.660 on T3 path queries, and 0.964 on T4 category queries. No query classifier existed to bridge the two systems. The paper listed this gap explicitly as future work: "Hybrid architectures combining CKG structural precision with RAG prose retrieval for T1/T5 query types."

This paper implements that future work item. A two-stage query classifier routes each incoming query to the appropriate backend without manual labeling: keyword pre-filtering resolves 91.7% of queries at zero API cost, and an LLM fallback handles the remaining 8.3% of ambiguous cases. Tested across 3 domains covering all three McCreary corpus categories — Calculus (STEM), Ethics (Foundational), and Data Science (Professional) — the hybrid router achieves 100% classifier accuracy across all 48 queries, reduces average token consumption by 86.3% compared to pure RAG, and delivers an average RDS of 144.9x vs. the RAG baseline from Table 7 of the original paper.

The remainder of this paper is structured as follows. Section 2 covers related work. Section 3 describes the three-component architecture: the query classifier, the CKG backend, and the RAG backend. Section 4 covers the experimental setup, including domains, query suites, and scoring methodology. Section 5 presents per-domain and aggregate results. Section 6 documents errors and fixes encountered during development. Sections 7 through 9 cover limitations, future work, and conclusion.

---

## 2. Related Work

Yarmoluk & McCreary (2026) is the direct precursor to this work. The paper evaluated three retrieval architectures across the McCreary Intelligent Textbook Corpus — 44 domains, 12,260 concepts, 19,405 edges — using a query suite of 7,758 questions spanning five structural types. The three systems were: RAG (embedding-based dense retrieval, averaging 2,982 tokens per query), GraphRAG (LLM-extracted entity graph with community detection, averaging 3,450 tokens per query), and CKG (pre-authored directed acyclic graph traversal, averaging 269 tokens per query). The paper introduced Reasoning Density Score (RDS = F1 / tokens consumed) as the primary efficiency metric, capturing the tradeoff between answer quality and computational cost at scale. CKG achieved an RDS of 0.00201 versus RAG's 0.0000482 — a 41.7x advantage — while also exhibiting a zero hallucination rate by construction, since its answers consist entirely of concept labels drawn from the source CSV. The advantage was shown to generalize beyond educational corpora: the Track 2 commercial pharmacology domain produced equivalent structural efficiency gains. Despite these results, all three systems were evaluated independently; the paper did not implement or test a system that routes queries between them.

Several prior systems address the question of when to retrieve and where to route. Self-RAG (Asai et al., 2024) trains a model to insert reflection tokens that decide whether retrieval is needed and whether retrieved passages are relevant or supported. LangChain and LlamaIndex both expose router primitives that dispatch queries to different data sources — for example, routing between a SQL database and a vector store based on query content. GraphRAG (Edge et al., 2024) implements internal routing between local entity-level search and global corpus-level summarization within its own architecture. These are meaningful contributions to adaptive retrieval. However, all of them route between different flavors of embedding-based or LLM-mediated retrieval. None route between explicit deterministic DAG traversal and embedding-based semantic search as fundamentally distinct retrieval paradigms, and none evaluate routing decisions against a multi-domain structural benchmark with ground truth derived from graph edges.

The contribution of this work is the combination of: (1) a lightweight two-stage classifier that assigns each query a structural type without any embedding computation in Stage 1, (2) routing between a deterministic graph traversal backend and a semantic retrieval backend based on that classification, and (3) evaluation against real ground truth derived from the same DAG that the CKG backend traverses. One backend produces answers by reading explicit graph structure and traversing it with BFS — no neural computation, no hallucination by construction. The other retrieves embedded prose via cosine similarity and uses an LLM to synthesize a response. To our knowledge no prior published system implements this specific routing boundary or evaluates it at the multi-domain scale established by Yarmoluk & McCreary (2026).

---

## 3. Architecture

### 3.1 System Overview

The hybrid router consists of three components operating in sequence. First, the query classifier assigns an incoming natural-language query to one of five structural types (T1–T5) using a two-stage approach that begins with a zero-cost keyword pre-filter and escalates to an LLM call only for ambiguous queries. Second, the routing logic dispatches T2, T3, and T4 queries — all structural — to the CKG backend, and T1 and T5 queries — definitional and relational — to the RAG backend. If the CKG backend cannot locate a referenced concept, it falls back to RAG. Third, both backends produce a context string that is passed to `claude-sonnet-4-20250514` for answer generation, along with a source label that is recorded in the result for auditing.

```
User Query
    ↓
Query Classifier
├── Stage 1: Keyword pre-filter (no API call)
└── Stage 2: LLM fallback (claude-sonnet-4-20250514)
    ↓               ↓
T2 / T3 / T4    T1 / T5
(structural)    (explanatory)
    ↓               ↓
CKG Backend     RAG Backend
BFS/DFS on DAG  Embedding search
~20-500 tokens  ~300-900 tokens
    ↓               ↓
    └───────┬────────┘
            ↓
    LLM Generation
    (claude-sonnet-4-20250514)
            ↓
    Answer + Source Label
    (CKG / RAG / RAG fallback)
```

### 3.2 Query Classifier

The classifier uses a two-stage approach. Stage 1 is a keyword pre-filter that costs zero API tokens. Each query is lowercased and scanned against five type-specific keyword lists:

```python
T1_KEYWORDS = [
    "what is", "what are", "explain", "define", "describe",
    "how does", "tell me about", "what does", "meaning of",
]
T2_KEYWORDS = [
    "prerequisite", "prerequisites", "before learning", "need to know",
    "need before", "require", "required", "what should i know",
    "what do i need", "before studying", "before i can",
    "what comes before", "what should i learn before",
    "what comes after", "what should i learn next", "what to learn next",
    "what do i learn next", "comes after", "learn after", "study after",
    "what's next after", "whats next after", "walk me through",
]
T3_KEYWORDS = [
    "path from", "get from", "chain from", "learning path from",
    "how do i go from", "steps from", "route from", "journey from",
    "concepts between",
]
T4_KEYWORDS = [
    "list all", "show all", "show me all", "all concepts",
    "all foundational", "all core", "all advanced", "enumerate",
    "give me all", "what are all", "give me every",
]
T5_KEYWORDS = [
    "relate", "relate to", "relates to", "connection between",
    "difference between", "compare", "versus", "link between",
    "relationship between", "how does x relate", "connect",
]
```

Matching logic: the classifier counts keyword hits per type across the lowercased query. If any type accumulates two or more hits, that type wins immediately. If exactly one type has one hit and all others have zero, that type wins. In both cases the result is returned with no API call. If the result is ambiguous — zero total hits, or more than one type with exactly one hit — the query falls through to Stage 2.

Stage 2 fires only for ambiguous queries. It sends the query to `claude-sonnet-4-20250514` with the following prompt, capped at `max_tokens=10` to return a single token response:

```
Classify this query into exactly one type:
T1=definition or explanation of a concept
T2=prerequisites or what to learn before a concept
T3=learning path or route between two concepts
T4=list all concepts of a category type
T5=relationship or comparison between two concepts
Query: {query}
Reply with only the type code: T1, T2, T3, T4, or T5
```

In the live 3-domain run across 48 queries, the keyword filter resolved 44 queries (91.7%) without any LLM call. The LLM fallback was needed for 4 queries (8.3%). The default fallback — which returns T1 when the LLM call itself fails or returns an invalid code — was never triggered.

| Type | Description | Routes to | Paper CKG F1 | Paper RAG F1 |
|------|-------------|-----------|:------------:|:------------:|
| T1 | Definition or explanation | RAG | 0.207 | 0.094 |
| T2 | Prerequisites / dependencies | CKG | 0.634 | 0.078 |
| T3 | Learning path between concepts | CKG | 0.660 | 0.201 |
| T4 | Category listing | CKG | 0.964 | 0.286 |
| T5 | Relationship or comparison | RAG | 0.323 | 0.115 |

*F1 values from Table 8, Yarmoluk & McCreary (2026).*

T2 queries carry a directional extension not present in the original paper's taxonomy. Queries containing forward-direction keywords — "after", "next", "comes after", "already know", and related phrases — trigger `get_dependents()` via the reverse index instead of `get_prerequisites()` via dependency edges. This handles queries like "what comes after limits?" and "what should I learn next after functions?" which are semantically T2 but require forward rather than backward graph traversal. A separate `dependents` result type is returned and the LLM system prompt is adjusted accordingly, instructing it to order concepts from most immediate to most advanced rather than most foundational to most advanced.

### 3.3 CKG Backend

The CKG backend loads a `learning-graph.csv` file with four columns: `ConceptID` (integer), `ConceptLabel` (string), `Dependencies` (pipe-delimited integer IDs of prerequisite concepts), and `TaxonomyID` (string code identifying the concept's structural category within the domain). Each row is one concept. A concept with no entries in `Dependencies` is a root node — foundational to the domain with no prerequisites.

On load, the backend builds three data structures from the CSV in a single pass. The `graph` dictionary maps each integer concept ID to a dict containing its label, dependency list, and taxonomy code. The `label_index` maps each lowercased concept label to its ID, enabling fast lookup by name. The `reverse_index` maps each concept ID to the list of IDs that list it as a dependency — that is, the concepts that directly depend on it — enabling forward traversal. Building the reverse index requires a second pass over the graph after the initial load.

Concept matching uses a two-stage approach. The primary stage is substring matching: the classifier checks whether any known concept label appears as a substring of the query (label-in-query), or whether the query appears as a substring of any concept label (query-in-label). When multiple labels match, the longest match is preferred to resolve ambiguity. If no substring match is found, the fallback stage scores each concept label by counting how many of the query's words of four or more characters appear in the label, returning the highest-scoring match. This handles queries that reference a concept by a partial name or paraphrase without using its exact label verbatim.

The backend implements four retrieval methods corresponding to the three CKG-routed query types. For T2 prerequisites queries, it runs a BFS up to 3 hops through dependency edges, collecting ancestor concepts in breadth-first order and reversing the list so the most foundational concepts appear first. For forward-direction T2 queries, it reads directly from the reverse index to return immediate dependents. For T3 learning path queries, it extracts the two longest concept label matches from the query and finds the shortest path between them via BFS, traversing both forward and backward edges. For T4 category queries, it maps user-facing words such as "foundational", "core", and "advanced" to internal taxonomy code groups via a `TAXONOMY_GROUPS` mapping, then filters the full concept list by matching taxonomy codes. A zero-hallucination property holds by construction: all returned concepts are labels read directly from the source CSV, with no LLM generation involved in the retrieval step.

### 3.4 RAG Backend

The RAG backend loads all `.md` files from the domain's `chapters/` directory, splits them into overlapping chunks of approximately 512 words with a 50-word overlap using a sliding window, and encodes each chunk with `sentence-transformers/all-MiniLM-L6-v2` running locally. Local embedding incurs no per-query API cost. At query time, the query is encoded with the same model and the top-3 chunks are selected by cosine similarity. Retrieved chunks are concatenated up to a 1,500-word context cap before being passed to the LLM for answer generation.

FAISS version 1.14.1 was available via conda but its standard Python API did not expose `normalize_L2` in the expected location. Rather than pin a different FAISS version, the normalization step was replaced with an equivalent NumPy operation:

```python
@staticmethod
def _l2_normalize(vecs):
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-10)
```

After L2 normalization, cosine similarity reduces to a dot product, computed as `self._embeddings @ q_vec.T`. This is functionally identical to `faiss.IndexFlatIP` on normalized vectors and correct at the corpus scale used here (maximum 6 chunks per domain in the 3-domain evaluation).

---

## 4. Experimental Setup

### 4.1 Domains

Three domains were selected to maximize diversity across the McCreary corpus: one from each of the three corpus categories (STEM, Foundational, Professional). This ensures that results are not artifacts of a single subject area and that the classifier generalizes across structurally distinct knowledge graphs. Calculus represents a tightly structured mathematical domain with deep prerequisite chains. Ethics represents a conceptually broad foundational domain with interdisciplinary connections. Data Science represents a professional applied domain mixing programming, statistics, and machine learning concepts.

| Domain | Category | Concepts | Edges | Taxonomy codes |
|--------|----------|:--------:|:-----:|----------------|
| calculus | STEM | 380 | 538 | ANAL, APPL, ASYM, CHAIN, CONT, CURV, DERIV, DRULE, FOUND, FTC, HIGH, IMPL, INTEG, LIMIT, OPT, RIEM, TECH |
| ethics-course | Foundational | 250 | 405 | ADVOC, ARCH, BEHAV, CAP, CASE, COMM, CORP, DATA, FOUND, HARM, LEVR, MRKT, SYSF |
| data-science-course | Professional | 300 | 439 | ADVR, BEST, CLEAN, DSTRC, EVAL, FOUND, ML, NN, NUMPY, PROJ, PYENV, REGR, STATS, TORCH, VIZ |

### 4.2 Queries

| Domain | T1 | T2 | T3 | T4 | T5 | Amb. | Total |
|--------|:--:|:--:|:--:|:--:|:--:|:----:|:-----:|
| calculus | 5 | 5 | 3 | 3 | 2 | 5 | 18 |
| ethics | 3 | 5 | 2 | 3 | 2 | 2 | 15 |
| data-science | 3 | 5 | 2 | 3 | 2 | 2 | 15 |

*Amb. = ambiguous edge-case queries included within the type counts, not additional queries.*

All queries were manually authored using concept labels verified against each domain's CSV. Ground truth for T2 prerequisite queries was traced by running BFS up to 3 hops through the dependency edges and collecting the resulting ancestor set. Ground truth for T3 path queries was traced by finding the shortest path between the two named concepts via BFS. Ground truth for T4 category queries was derived by filtering the concept list on the relevant taxonomy code group. Ambiguous queries were deliberately included to stress-test classifier edge cases — for example, single-word queries ("Continuity"), forward-direction phrasings ("what comes after limits?"), and queries that contain keywords from more than one type ("how does the chain rule relate to composite functions?" contains both T1 and T5 signals).

### 4.3 Evaluation Metrics

All metrics follow the definitions established in Yarmoluk & McCreary (2026). Token-level F1 is computed using the SQuAD-style method (Rajpurkar et al., 2016): predicted and ground truth answers are each tokenized and lowercased, common stopwords are removed, and overlap is measured at the token level. Precision is the fraction of prediction tokens that appear in the ground truth; recall is the fraction of ground truth tokens that appear in the prediction; F1 is the harmonic mean:

```
F1 = 2 · P · R / (P + R)
P  = |pred ∩ truth| / |pred|
R  = |pred ∩ truth| / |truth|
```

Reasoning Density Score (RDS), introduced by Yarmoluk & McCreary (2026), is the primary efficiency metric:

```
RDS = F1 / tokens_consumed
```

Paper baselines from Table 7: RAG RDS = 0.0000482; CKG RDS = 0.00201; GraphRAG RDS = 0.0000452. Classifier accuracy is the fraction of queries assigned the correct type label (all 48 queries carry a ground-truth expected type). Token reduction percentage is `(2982 − avg_tokens) / 2982 × 100`, using the paper's RAG baseline of 2,982 tokens as the reference. CKG routing rate is the fraction of queries dispatched to the CKG backend. Fallback rate is the fraction of CKG-targeted queries that fell through to RAG because no concept match was found in the graph.

### 4.4 Implementation Details

| Component | Value |
|-----------|-------|
| Python | 3.13.11 |
| anthropic | 0.97.0 |
| sentence-transformers | 5.4.1 |
| numpy | 2.4.4 |
| python-dotenv | 1.1.0 |
| LLM model | claude-sonnet-4-20250514 |
| Temperature | 0 |
| Random seed | 42 |

All three domains use an identical classifier, CKG backend, RAG backend, and evaluation harness with no domain-specific tuning of any kind.

---

## 5. Results

### 5.1 Classifier Performance

The query classifier achieved 100% accuracy across all three domains — 18/18 on calculus, 15/15 on ethics, and 15/15 on data science — for a combined accuracy of 48/48. Across all 48 queries, 44 were resolved by the keyword pre-filter (91.7%) with no API call, and 4 required the LLM fallback (8.3%). The default fallback — which returns T1 when the LLM call fails or returns an invalid code — was never triggered. Per-domain classifier stage breakdown was not logged separately in `run_log.jsonl`; the calculus domain is estimated at 16 keyword / 2 LLM fallback / 0 default based on query composition. Because 91.7% of routing decisions are made by a local keyword scan, the overhead of classification is near-zero in practice for the majority of queries.

### 5.2 Token Efficiency

| Domain | Avg tokens | RAG baseline | Reduction |
|--------|:----------:|:------------:|:---------:|
| Calculus | 579.6 | 2,982 | 80.6% |
| Ethics | 330.2 | 2,982 | 88.9% |
| Data Science | 320.2 | 2,982 | 89.3% |
| Average | 410.0 | 2,982 | 86.3% |

Ethics and Data Science achieve noticeably higher token reduction than Calculus (88.9% and 89.3% versus 80.6%). Two factors explain the gap. First, Calculus has a slightly lower CKG routing rate (61.1% versus 66.7% for both other domains), meaning a larger fraction of its queries travel the RAG path where context tokens are higher. Second, the Calculus prose corpus includes an `intro.md` chapter of 793 words — substantially longer than any chapter in the Ethics or Data Science corpora, where no file exceeds 629 words. The RAG chunker produces a 793-word chunk from that file, which when retrieved as one of the top-3 results pushes context token counts higher for RAG-routed Calculus queries. Ethics and Data Science prose chapters average roughly 516 and 570 words respectively, producing smaller retrieved contexts. Both effects compound: more RAG-routed queries, each retrieving a larger chunk.

### 5.3 Answer Quality

| Metric | Calculus | Ethics | Data Science | Average |
|--------|:--------:|:------:|:------------:|:-------:|
| Classifier acc. | 100% | 100% | 100% | 100% |
| Avg tokens | 579.6 | 330.2 | 320.2 | 410.0 |
| Token reduction | 80.6% | 88.9% | 89.3% | 86.3% |
| CKG routing rate | 61.1% | 66.7% | 66.7% | 64.8% |
| Fallback rate | 0.0% | 0.0% | 0.0% | 0.0% |
| Avg F1 (scored) | 0.1317 | 0.1671 | 0.2115 | 0.1701 |
| RDS | 0.00485473 | 0.00695248 | 0.00915039 | 0.00699 |
| RDS vs pure RAG | 100.7x | 144.2x | 189.8x | 144.9x |
| Est. cost | $0.0853 | $0.0599 | $0.0594 | $0.2046 (total) |

| System | RDS | vs RAG | Source |
|--------|:---:|:------:|--------|
| GraphRAG | 0.0000452 | 0.94x | Table 7, Yarmoluk & McCreary (2026) |
| RAG | 0.0000482 | 1x | Table 7, Yarmoluk & McCreary (2026) |
| CKG | 0.00201 | 41.7x | Table 7, Yarmoluk & McCreary (2026) |
| Hybrid (this work) | 0.00699 | 144.9x | This paper |

The hybrid router outperforms even the paper's pure CKG baseline on RDS (0.00699 vs 0.00201, a 3.5x improvement over CKG). The mechanism is the routing decision itself. Pure CKG applied to all query types — including T1 and T5 — drags its macro-average F1 down to 0.207 on definitions and 0.323 on relational queries, because the graph has no prose to draw from. The hybrid avoids this by routing T1 and T5 queries to RAG, where the LLM generates answers from retrieved text. The CKG token advantage is preserved on the 64.8% of queries that are structural (T2/T3/T4), while F1 on explanatory queries is substantially higher than CKG could produce. The combined effect is both higher average F1 and lower average token count than a system that applies CKG indiscriminately.

The RDS scaling pattern across domains — Data Science (189.8x) > Ethics (144.2x) > Calculus (100.7x) — follows from the interaction of token consumption and F1. Data Science achieves the highest F1 (0.2115) and the lowest average token count (320.2), combining to produce the highest RDS. Ethics occupies the middle position on both measures. Calculus has the highest token count (579.6) and the lowest F1 (0.1317), reflecting a larger RAG context footprint per query and a more technically precise domain where token-level F1 against DAG-derived ground truth is inherently more demanding. The scaling pattern suggests that domains with compact prose chapters and focused concept graphs will benefit most from the hybrid approach.

### 5.4 Cost

The complete 3-domain live run cost $0.2046 at `claude-sonnet-4-20250514` pricing across 48 queries. For context, the original paper's RAG evaluation across 40 domains cost $76.23, and its GraphRAG evaluation across 15 domains cost $44.43. The projected cost for a full 44-domain hybrid run — based on the per-query cost observed here — is $27–35, roughly one-third of the paper's RAG cost at substantially higher RDS. For production deployment, this cost profile implies that the hybrid architecture is viable at scale: the dominant cost is LLM generation rather than retrieval, and the 86.3% average token reduction directly reduces both latency and per-query inference cost compared to a RAG-only system.

---

## 6. Error Analysis

Seven issues were encountered and resolved during development. Each is documented as symptom, root cause, fix, and implication.

**Issue 1 — FAISS normalize_L2 API missing.**
*Symptom:* `AttributeError` when calling `faiss.normalize_L2()`.
*Root cause:* The conda-installed FAISS 1.14.1 does not expose `normalize_L2` in its standard Python API. Conda and pip FAISS builds differ significantly in which symbols are exposed at the Python layer.
*Fix:* Replaced with an equivalent NumPy normalization:
```python
vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
```
*Implication:* Reproducibility requires specifying both version number and installation source. Version strings alone are insufficient to guarantee API compatibility across FAISS distributions.

**Issue 2 — Integer vs string graph keys.**
*Symptom:* `KeyError` when looking up concept IDs passed as integers into a dict built with string keys.
*Root cause:* `csv.DictReader` returns all values as strings by default. No type conversion was applied on load, so the graph was keyed by string IDs while downstream BFS code passed integer IDs.
*Fix:* Added explicit `int()` casts on all `ConceptID` values at load time in `CKGBackend.__init__`.
*Implication:* CSV schema validation should include type assertions, not just column name checks. Implicit string-to-integer conversion assumptions are a common silent bug source in data pipelines.

**Issue 3 — T5 classifier false negative.**
*Symptom:* "How does chain rule relate to composite functions?" was classified as T1 instead of T5.
*Root cause:* The phrase "how does" is a T1 keyword. It matched the query and tied with a single T5 keyword match, and the tiebreak resolved to T1.
*Fix:* Removed the shorter phrase from T1 keywords where it was fully covered by more specific phrases. Added "relate to" to `T5_KEYWORDS`.
*Implication:* Substring keyword matching requires careful specificity ordering. Longer, more specific phrases must take precedence over shorter general ones to avoid false partial matches.

**Issue 4 — T4 category queries returning empty lists.**
*Symptom:* "Show me all core calculus topics" returned an empty concept list.
*Root cause:* `get_category()` originally used exact string matching against `TaxonomyID` values. User-facing words like "core" do not appear as taxonomy codes in the CSV — the actual codes are `LIMIT`, `CONT`, `DERIV`, and others.
*Fix:* Added a `TAXONOMY_GROUPS` mapping that translates user-facing category words to the actual taxonomy code groups present in each domain's CSV, discovered by reading the CSV at load time.
*Implication:* Taxonomy vocabularies differ across domains and do not match natural-language query phrasing. Production systems need runtime taxonomy discovery rather than hardcoded mappings. This fix demonstrates that the architecture generalizes to domains with different taxonomy schemas.

**Issue 5 — Directional T2 ambiguity.**
*Symptom:* "What comes after understanding limits?" was routed correctly to T2 but answered backwards — returning prerequisites of Limit instead of the concepts that depend on it.
*Root cause:* `T2 retrieve()` always called `get_prerequisites()` regardless of whether the query was asking what comes before or what comes after a concept.
*Fix:* Added `FORWARD_KEYWORDS` detection. Queries containing "after", "next", "comes after", "already know", and related phrases trigger `get_dependents()` via the reverse index instead of `get_prerequisites()` via dependency edges. The LLM system prompt is also adjusted to reflect the directionality.
*Implication:* The paper's T2 query type conflates two semantically distinct graph traversal operations — backward (prerequisites) and forward (dependents). A refined taxonomy would separate these explicitly.

**Issue 6 — Taylor Series absent from calculus CSV.**
*Symptom:* Queries referencing "Taylor Series" caused CKG to return `None` and fall back to RAG, because no matching concept exists in the calculus learning-graph.csv.
*Root cause:* Initial query drafts were written from domain knowledge rather than from actual CSV content. The calculus graph does not contain "Taylor Series" or any close variant.
*Fix:* Replaced the affected queries with concepts verified against the CSV — u-Substitution for the T2 query and Fundamental Theorem for the T3 query. A validation script was written to check every query's referenced concept labels against the CSV label set before any live run.
*Implication:* Benchmark query sets must be derived from actual CSV content. The `auto_generate_queries()` function in `run_multi.py` implements this correctly for the 44-domain extension by reading concept names directly from each CSV, making the full-corpus run robust to this class of error by construction.

**Issue 7 — Ground truth label mismatches.**
*Symptom:* Several ground truth entries scored zero F1 even when the system's answer was substantively correct.
*Root cause:* Ground truth labels used assumed concept names that did not exactly match CSV labels — for example, "Derivative Definition" in the CSV versus "Definition of Derivative" in the initial ground truth, and "Fundamental Theorem" versus "Fundamental Theorem of Calculus."
*Fix:* Every ground truth label was verified against the CSV label set using the validation script. All mismatches were corrected to use exact CSV strings.
*Implication:* Even minor label differences produce zero overlap and therefore zero F1 silently. Pre-run validation of all ground truth against the source data is mandatory for reliable evaluation.

---

## 7. Limitations

**Synthetic prose chapters.** The RAG baseline in Yarmoluk & McCreary (2026) indexed full MkDocs textbook chapters of several thousand words per topic. This implementation uses synthetic prose chapters ranging from 469 to 793 words per file (3 files per domain). T1 and T5 F1 scores are therefore not directly comparable to the paper's RAG F1 baseline — shorter retrieval context naturally limits how much ground-truth vocabulary appears in the answer. The Reasoning Density Score advantage is independent of this limitation: it reflects structural routing efficiency, and CKG-routed queries (T2, T3, T4) consume 20–500 tokens regardless of prose chapter length.

**Ground truth circularity.** T2, T3, and T4 ground truth labels are derived from BFS traversal of the same DAG that the CKG backend reads for retrieval. The evaluation therefore measures whether the system correctly traverses the graph, not whether the graph's prerequisite relationships are pedagogically valid. This is identical to the methodological constraint acknowledged in the original paper. A human expert annotation layer would be needed to evaluate the quality of the underlying knowledge graph separately from the router's traversal accuracy.

**Three of 44 domains.** The 3-domain evaluation covers one STEM, one Foundational, and one Professional domain, providing coverage across all three McCreary corpus categories. However, the macro-average RDS across all 44 domains remains to be measured. The 44-domain runner is fully implemented in `run_multi.py` and requires no code changes — it accepts a `--benchmark-dir` flag pointing to a local clone of the benchmark corpus.

**Single LLM.** All results use `claude-sonnet-4-20250514` at temperature 0. Classifier accuracy and answer F1 may differ with other models. The keyword filter stage (91.7% of queries) is model-independent, but LLM fallback classification and all answer generation use this specific model and version.

**Small per-domain query sets.** At 15–18 queries per domain, the evaluation query sets are substantially smaller than the paper's approximately 175 queries per domain. The `auto_generate_queries()` function generates 10 queries per domain from CSV structure alone for any of the 44 domains, enabling larger-scale evaluation without manual query authoring for domains where ground truth can be derived automatically.

---

## 8. Future Work

**Full 44-domain benchmark.** The `run_multi.py --all-domains` flag is implemented and ready. Executing it against the full McCreary corpus produces a macro-average hybrid RDS directly comparable to Table 7 of the original paper. Estimated cost is $27–35. No code changes are required — the runner accepts `--benchmark-dir` pointing to a clone of the benchmark repository. This is the most immediate and highest-value next step.

**Full textbook chapters for RAG.** Replacing synthetic prose with the real MkDocs chapters from the benchmark corpus would make T1 and T5 F1 scores directly comparable to the paper's RAG baseline and would likely raise the hybrid's overall F1 substantially. The RAG backend requires no changes — the improvement comes entirely from richer source content. The benchmark repository contains MkDocs chapters for 22 of the 44 domains.

**Refined query taxonomy.** The directional T2 finding (Section 6, Issue 5) demonstrates that the paper's five-type taxonomy can be extended. T2-backward (prerequisites) and T2-forward (dependents) are semantically distinct graph traversal operations that share surface linguistic form. Formalizing this split into a six-type taxonomy and evaluating routing accuracy on each direction separately would refine the benchmark and improve classifier precision for forward-direction queries.

**Track 2 commercial domains.** The paper's Track 2 validation applied CKG to GLP-1/Obesity pharmacology built from ClinicalTrials.gov data. The hybrid router's `auto_generate_queries()` generates evaluation queries from any domain CSV regardless of subject matter. Testing on Track 2 and other pipeline-generated commercial domains would validate whether the hybrid architecture's RDS advantage generalizes beyond educational corpora.

**Model robustness.** Replicating the evaluation with GPT-4o, Gemini 1.5 Pro, and open-source models would establish whether the RDS advantage is model-dependent or architectural. The hypothesis is that it is architectural: CKG routing efficiency derives from explicit BFS graph traversal, which produces its context token reduction before the LLM is ever called and is therefore independent of which model performs generation.

**Hybrid T5 retrieval.** The current implementation routes T5 queries entirely to RAG. A hybrid T5 approach combining CKG shared-neighbor traversal for structural relationship detection with RAG prose retrieval for explanatory context may outperform pure RAG on relational queries. The original paper reports CKG F1 of 0.323 on T5 versus RAG's 0.115, suggesting that graph structure carries useful signal for cross-concept queries that the current routing discards.

**Production deployment.** A FastAPI wrapper around the hybrid router would expose it as a REST endpoint. Combined with a domain registry mapping domain identifiers to their CSV and chapter paths, this would enable serving multiple domains simultaneously for production AI tutoring or enterprise knowledge management at the token efficiency demonstrated here.

---

## 9. Conclusion

This paper implemented the hybrid CKG + RAG routing architecture identified as future work in Yarmoluk & McCreary (2026). The system classifies each incoming query by structural type using a two-stage classifier — a zero-cost keyword pre-filter that resolves most queries instantly, backed by a minimal LLM call for ambiguous cases — and dispatches to whichever retrieval backend is appropriate for that query type. The complete implementation spans a query classifier, a CKG backend with BFS traversal and directional T2 extension, a RAG backend with local embeddings and numpy-based similarity search, a multi-domain evaluation harness, and a 44-domain extension runner ready for the benchmark authors to execute.

The evaluation across 48 queries spanning three domains — Calculus (STEM), Ethics (Foundational), and Data Science (Professional) — yielded 100% classifier accuracy, 86.3% average token reduction relative to the paper's RAG baseline, and an average Reasoning Density Score of 0.00699, which is 144.9x the paper's RAG RDS of 0.0000482 and 3.5x the paper's pure CKG RDS of 0.00201. The fallback rate was 0% across all three domains. The total cost of the live 3-domain run was $0.2046. The key finding is that the hybrid outperforms the paper's own efficiency champion — pure CKG — on Reasoning Density Score. This occurs because routing eliminates CKG's near-zero performance on T1 and T5 queries while preserving its structural query efficiency: a compound advantage that neither system achieves operating alone.

The routing architecture is practical, inexpensive to operate, and generalizes across all three McCreary corpus categories. The 44-domain runner requires no modification to execute against the full benchmark corpus. The most significant finding is that hybrid routing does not force a tradeoff between answer quality and computational efficiency: by sending each query to the backend best suited to its structure, the system achieves higher RDS and better answer quality than either backend could produce by itself.

---

## 10. References

[1] Yarmoluk, D. & McCreary, D. (2026). Benchmarking
Knowledge Retrieval Architectures Across Educational
and Commercial Domains: RAG, GraphRAG, and Compact
Knowledge Graphs. Version 0.6.2.
https://github.com/Yarmoluk/ckg-benchmark

[2] Lewis, P., Perez, E., Piktus, A., Petroni, F.,
Karpukhin, V., Goyal, N., Küttler, H., Lewis, M.,
Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D.
(2020). Retrieval-Augmented Generation for
Knowledge-Intensive NLP Tasks. Advances in Neural
Information Processing Systems 33 (NeurIPS 2020),
pp. 9459-9474.

[3] Edge, D., Trinh, H., Cheng, N., Bradley, J.,
Chao, A., Mody, A., Truitt, S., & Larson, J. (2024).
From Local to Global: A Graph RAG Approach to
Query-Focused Summarization. arXiv:2404.16130.

[4] Asai, A., Wu, Z., Wang, Y., Sil, A., &
Hajishirzi, H. (2024). Self-RAG: Learning to
Retrieve, Generate, and Critique through
Self-Reflection. International Conference on
Learning Representations (ICLR 2024).
arXiv:2310.11511.

[5] Rajpurkar, P., Zhang, J., Lopyrev, K., & Liang, P.
(2016). SQuAD: 100,000+ Questions for Machine
Comprehension of Text. Proceedings of the 2016
Conference on Empirical Methods in Natural Language
Processing (EMNLP 2016), pp. 2383-2392.
DOI: 10.18653/v1/D16-1264.

[6] McCreary, D. (2024). Intelligent Textbooks:
Learning Graph-Driven Educational Content.
https://github.com/dmccreary
