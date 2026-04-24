STOPWORDS = {"the", "a", "an", "is", "are", "of", "to", "in", "and", "for", "this", "that"}


def token_f1(prediction_text, ground_truth_list):
    if ground_truth_list is None or len(ground_truth_list) == 0:
        return None
    if not prediction_text:
        return 0.0

    truth_str = " ".join(ground_truth_list)
    pred_tokens = {t for t in prediction_text.lower().split() if t not in STOPWORDS}
    truth_tokens = {t for t in truth_str.lower().split() if t not in STOPWORDS}

    if not pred_tokens or not truth_tokens:
        return 0.0

    overlap = pred_tokens & truth_tokens
    if not overlap:
        return 0.0

    precision = len(overlap) / len(pred_tokens)
    recall = len(overlap) / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_rds(f1, tokens):
    if tokens == 0 or f1 is None:
        return 0.0
    return round(f1 / tokens, 8)


def classifier_accuracy(results):
    valid = [r for r in results if r.get("expected_type") is not None]
    if not valid:
        return 0.0
    correct = sum(1 for r in valid if r.get("query_type") == r.get("expected_type"))
    return correct / len(valid)


def routing_stats(results):
    if not results:
        return {}

    ckg = [r for r in results if r.get("source") == "CKG"]
    rag = [r for r in results if r.get("source") == "RAG"]
    fallback = [r for r in results if "fallback" in r.get("source", "")]
    scored = [r for r in results if r.get("f1") is not None]

    def mean(values):
        return sum(values) / len(values) if values else 0

    avg_tokens_ckg = mean([r["context_tokens"] for r in ckg])
    avg_tokens_rag = mean([r["context_tokens"] for r in rag])
    avg_tokens_all = mean([r["context_tokens"] for r in results])
    tokens_saved_pct = (2982 - avg_tokens_all) / 2982 * 100

    avg_f1 = mean([r["f1"] for r in scored])
    rds_scores = [compute_rds(r["f1"], r["context_tokens"]) for r in scored]
    avg_rds = mean(rds_scores)

    return {
        "total_queries": len(results),
        "ckg_routed": len(ckg),
        "rag_routed": len(rag),
        "fallback_count": len(fallback),
        "ckg_routing_rate_pct": len(ckg) / len(results) * 100,
        "avg_tokens_ckg": round(avg_tokens_ckg, 1),
        "avg_tokens_rag": round(avg_tokens_rag, 1),
        "avg_tokens_hybrid": round(avg_tokens_all, 1),
        "tokens_saved_pct": round(tokens_saved_pct, 1),
        "avg_f1_scored": round(avg_f1, 4),
        "avg_rds_hybrid": round(avg_rds, 8),
        "paper_rds_rag": 0.0000482,
        "paper_rds_ckg": 0.00201,
        "rds_vs_paper_rag": round(avg_rds / 0.0000482, 1) if avg_rds > 0 else 0,
        "total_estimated_cost": round(
            sum(r["estimated_cost"] for r in results), 4
        ),
    }
