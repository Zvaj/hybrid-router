import sys
import json
import os
import argparse
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from backends.ckg import CKGBackend
from backends.rag import RAGBackend
from router.classifier import classify_query, CLASSIFIER_STATS
from router.hybrid import route, MODEL_SHORTCUTS, DEFAULT_MODEL
from evaluation.queries import TEST_QUERIES
from evaluation.score import (token_f1, compute_rds,
                               classifier_accuracy, routing_stats)


def parse_args():
    parser = argparse.ArgumentParser(description="Hybrid CKG + RAG Router")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip all Anthropic API calls")
    parser.add_argument("--domain", type=str, default="calculus",
                        help="Which domain to use")
    parser.add_argument("--no-cache", action="store_true",
                        help="Bypass result cache")
    parser.add_argument(
        '--model',
        default='sonnet',
        choices=['haiku', 'sonnet', 'opus'],
        help='Model tier: haiku, sonnet, or opus'
    )
    return parser.parse_args()


def print_header(domain, dry_run, model_short, model_full):
    print("╔══════════════════════════════════════════╗")
    print("║      HYBRID CKG + RAG ROUTER             ║")
    print("║  Yarmoluk & McCreary (2026) — Future Work║")
    print("╚══════════════════════════════════════════╝")
    print(f"Domain: {domain} | Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print(f"Model: {model_short} ({model_full})")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


def print_query_result(result, expected_type, f1):
    correct = "✓" if result["query_type"] == expected_type else "✗"
    f1_str = f"{f1:.4f}" if f1 is not None else "N/A"
    print("┌─────────────────────────────────────────────┐")
    print(f"│ Query: {result['query']}")
    print(f"│ Expected: {expected_type} | Got: {result['query_type']} {correct}")
    print(f"│ Routed to: {result['source']}")
    print(f"│ Context tokens: {result['context_tokens']} "
          f"(saved {result['tokens_saved']} vs RAG baseline)")
    print(f"│ Est. cost: ${result['estimated_cost']:.5f}")
    print(f"│ F1: {f1_str}")
    print("├─────────────────────────────────────────────┤")
    print(f"│ {result['answer'][:300]}")
    print("└─────────────────────────────────────────────┘")
    print()


def print_summary(stats, classifier_acc):
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                 HYBRID ROUTER RESULTS                       ║")
    print("╠══════════════════════╦═══════════╦══════════╦═══════════════╣")
    print("║ Metric               ║  Hybrid   ║ RAG*     ║ CKG*          ║")
    print("╠══════════════════════╬═══════════╬══════════╬═══════════════╣")
    print(f"║ Classifier accuracy  ║ {classifier_acc*100:>7.1f}%  ║          ║               ║")
    print(f"║ Avg tokens/query     ║ {stats['avg_tokens_hybrid']:>9.1f} ║ {2982:>8} ║ {269:>13} ║")
    print(f"║ Token reduction %    ║ {stats['tokens_saved_pct']:>7.1f}%  ║          ║               ║")
    print(f"║ CKG routing rate %   ║ {stats['ckg_routing_rate_pct']:>7.1f}%  ║          ║               ║")
    print(f"║ Fallback rate %      ║ {stats['fallback_count']/stats['total_queries']*100:>7.1f}%  ║          ║               ║")
    print(f"║ Avg F1 (scored)      ║ {stats['avg_f1_scored']:>9.4f} ║   0.1231 ║        0.4709 ║")
    print(f"║ RDS                  ║ {stats['avg_rds_hybrid']:>9.8f} ║  4.82e-5 ║       0.00201 ║")
    print(f"║ RDS vs pure RAG      ║ {stats['rds_vs_paper_rag']:>8.1f}x  ║     1.0x ║        41.7x  ║")
    print(f"║ Total est. cost      ║ ${stats['total_estimated_cost']:>8.4f} ║          ║               ║")
    print("╚══════════════════════╩═══════════╩══════════╩═══════════════╝")
    print("* Paper baselines from Table 7, Yarmoluk & McCreary (2026)")
    print(
        f"Classifier: {CLASSIFIER_STATS['keyword_filter']} keyword, "
        f"{CLASSIFIER_STATS['llm_fallback']} LLM fallback, "
        f"{CLASSIFIER_STATS['default']} default"
    )


def main():
    args = parse_args()
    model_name = MODEL_SHORTCUTS.get(args.model, DEFAULT_MODEL)
    print_header(args.domain, args.dry_run, args.model, model_name)

    base = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base, "data", args.domain, "learning-graph.csv")
    chapters_dir = os.path.join(base, "data", args.domain, "chapters")

    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found")
        sys.exit(1)

    import time
    t = time.time()
    ckg = CKGBackend(csv_path)
    print(f"CKG loaded in {time.time()-t:.1f}s")

    t = time.time()
    rag = RAGBackend(chapters_dir)
    print(f"RAG built in {time.time()-t:.1f}s")
    print()

    results = []
    for test in TEST_QUERIES:
        query = test["query"]
        expected = test["expected_type"]
        ground_truth = test.get("ground_truth")

        query_type = classify_query(
            query,
            use_llm_fallback=not args.dry_run,
            model=model_name
        )
        result = route(
            query, query_type, ckg, rag,
            args.dry_run,
            model=model_name
        )
        result["expected_type"] = expected

        f1 = token_f1(result["answer"], ground_truth)
        result["f1"] = f1

        print_query_result(result, expected, f1)
        results.append(result)

    stats = routing_stats(results)
    acc = classifier_accuracy(results)
    print_summary(stats, acc)

    output = {
        "run_info": {
            "domain": args.domain,
            "mode": "dry_run" if args.dry_run else "live",
            "timestamp": datetime.now().isoformat(),
            "total_queries": len(results),
        },
        "summary": stats,
        "classifier_accuracy": acc,
        "classifier_stats": CLASSIFIER_STATS,
        "queries": results,
    }

    results_file = f"results_{args.domain}_{args.model}.json"
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {results_file}")

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "domain": args.domain,
        "mode": "dry_run" if args.dry_run else "live",
        **stats,
        "classifier_accuracy": acc,
    }
    with open("run_log.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    print("Run logged to run_log.jsonl")


if __name__ == "__main__":
    main()
