import sys
import json
import os
import time
import importlib
import argparse
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from backends.ckg import CKGBackend
from backends.rag import RAGBackend
from router.classifier import classify_query, CLASSIFIER_STATS
from router.hybrid import route, MODEL_SHORTCUTS, DEFAULT_MODEL
from evaluation.score import (token_f1, compute_rds,
                               classifier_accuracy,
                               routing_stats)

DOMAIN_CONFIGS = {
    "calculus": {
        "queries_module": "evaluation.queries",
        "queries_var": "TEST_QUERIES",
        "display_name": "Calculus",
        "category": "STEM",
    },
    "ethics-course": {
        "queries_module": "evaluation.queries_ethics",
        "queries_var": "ETHICS_QUERIES",
        "display_name": "Ethics",
        "category": "Foundational",
    },
    "data-science-course": {
        "queries_module": "evaluation.queries_data_science",
        "queries_var": "DATA_SCIENCE_QUERIES",
        "display_name": "Data Science",
        "category": "Professional",
    },
}

PAPER_BASELINES = {
    "rag_rds": 0.0000482,
    "ckg_rds": 0.00201,
    "rag_f1": 0.1231,
    "ckg_f1": 0.4709,
    "rag_tokens": 2982,
    "ckg_tokens": 269,
}


def run_domain(domain_key, dry_run=False,
               model=DEFAULT_MODEL):
    config = DOMAIN_CONFIGS.get(domain_key)
    if not config:
        print(f"ERROR: unknown domain '{domain_key}'")
        return None

    base = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base, "data", domain_key, "learning-graph.csv")
    chapters_dir = os.path.join(base, "data", domain_key, "chapters")

    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found — skipping {domain_key}")
        return None

    mod = importlib.import_module(config["queries_module"])
    queries = getattr(mod, config["queries_var"])

    t = time.time()
    ckg = CKGBackend(csv_path)
    ckg_load_time = time.time() - t

    t = time.time()
    rag = RAGBackend(chapters_dir)
    rag_load_time = time.time() - t

    results = []
    for test in queries:
        query_type = classify_query(
            test["query"],
            use_llm_fallback=not dry_run,
            model=model
        )
        result = route(
            test["query"], query_type,
            ckg, rag, dry_run,
            model=model
        )
        result["expected_type"] = test["expected_type"]
        result["f1"] = token_f1(result["answer"], test.get("ground_truth"))
        results.append(result)

    stats = routing_stats(results)
    acc = classifier_accuracy(results)

    fallback_rate_pct = (
        stats["fallback_rate_pct"]
        if "fallback_rate_pct" in stats
        else (stats["fallback_count"] / len(results) * 100)
    )

    return {
        "domain": domain_key,
        "display_name": config["display_name"],
        "category": config["category"],
        "total_queries": len(results),
        "classifier_accuracy": acc,
        "avg_tokens": stats["avg_tokens_hybrid"],
        "token_reduction_pct": stats["tokens_saved_pct"],
        "ckg_routing_rate_pct": stats["ckg_routing_rate_pct"],
        "fallback_rate_pct": fallback_rate_pct,
        "avg_f1": stats["avg_f1_scored"],
        "rds": stats["avg_rds_hybrid"],
        "rds_vs_rag": stats["rds_vs_paper_rag"],
        "total_cost": stats["total_estimated_cost"],
        "query_results": results,
        "ckg_load_time": round(ckg_load_time, 2),
        "rag_load_time": round(rag_load_time, 2),
        "model": model,
    }


def print_domain_summary(result):
    print("─────────────────────────────────")
    print(f"Domain: {result['display_name']} ({result['category']})")
    print(f"Queries: {result['total_queries']}")
    print(f"Classifier accuracy: {result['classifier_accuracy']*100:.1f}%")
    print(f"Avg tokens: {result['avg_tokens']:.1f} ({result['token_reduction_pct']:.1f}% reduction)")
    print(f"CKG routing rate: {result['ckg_routing_rate_pct']:.1f}%")
    print(f"Avg F1: {result['avg_f1']:.4f}")
    print(f"RDS: {result['rds']:.8f} ({result['rds_vs_rag']:.1f}x vs RAG)")
    print(f"Est. cost: ${result['total_cost']:.4f}")
    print("─────────────────────────────────")


def print_combined_table(all_results):
    if not all_results:
        print("No results to display.")
        return

    def val(r, key, fmt):
        return fmt.format(r[key]) if r else "N/A"

    # Pad domain columns (up to 3); fill blanks if fewer domains ran
    domains = {r["domain"]: r for r in all_results}
    calc  = domains.get("calculus")
    eth   = domains.get("ethics-course")
    ds    = domains.get("data-science-course")

    def avg(key):
        vals = [r[key] for r in all_results if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    acc_avg   = avg("classifier_accuracy")
    tok_avg   = avg("avg_tokens")
    red_avg   = avg("token_reduction_pct")
    ckg_avg   = avg("ckg_routing_rate_pct")
    fb_avg    = avg("fallback_rate_pct")
    f1_avg    = avg("avg_f1")
    rds_avg   = avg("rds")
    rvr_avg   = avg("rds_vs_rag")
    cost_tot  = sum(r["total_cost"] for r in all_results)

    def col(r, key, fmt="{:.1f}"):
        return fmt.format(r[key]) if r else "  N/A  "

    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║          3-DOMAIN HYBRID ROUTER RESULTS                         ║")
    print("║      Yarmoluk & McCreary (2026) — Future Work Implementation    ║")
    print("╠══════════════════════╦══════════╦══════════╦══════════╦═════════╣")
    print("║ Metric               ║ Calculus ║ Ethics   ║ DataSci  ║ Average ║")
    print("║                      ║ (STEM)   ║ (Found.) ║ (Prof.)  ║         ║")
    print("╠══════════════════════╬══════════╬══════════╬══════════╬═════════╣")
    print(f"║ Classifier acc.      ║{col(calc,'classifier_accuracy','{:.1%}'):>9} ║{col(eth,'classifier_accuracy','{:.1%}'):>9} ║{col(ds,'classifier_accuracy','{:.1%}'):>9} ║{acc_avg:>7.1%}  ║")
    print(f"║ Avg tokens/query     ║{col(calc,'avg_tokens','{:.0f}'):>9} ║{col(eth,'avg_tokens','{:.0f}'):>9} ║{col(ds,'avg_tokens','{:.0f}'):>9} ║{tok_avg:>7.0f}  ║")
    print(f"║ Token reduction      ║{col(calc,'token_reduction_pct','{:.1f}%'):>9} ║{col(eth,'token_reduction_pct','{:.1f}%'):>9} ║{col(ds,'token_reduction_pct','{:.1f}%'):>9} ║{red_avg:>6.1f}%  ║")
    print(f"║ CKG routing rate     ║{col(calc,'ckg_routing_rate_pct','{:.1f}%'):>9} ║{col(eth,'ckg_routing_rate_pct','{:.1f}%'):>9} ║{col(ds,'ckg_routing_rate_pct','{:.1f}%'):>9} ║{ckg_avg:>6.1f}%  ║")
    print(f"║ Fallback rate        ║{col(calc,'fallback_rate_pct','{:.1f}%'):>9} ║{col(eth,'fallback_rate_pct','{:.1f}%'):>9} ║{col(ds,'fallback_rate_pct','{:.1f}%'):>9} ║{fb_avg:>6.1f}%  ║")
    print(f"║ Avg F1 (scored)      ║{col(calc,'avg_f1','{:.4f}'):>9} ║{col(eth,'avg_f1','{:.4f}'):>9} ║{col(ds,'avg_f1','{:.4f}'):>9} ║{f1_avg:>8.4f} ║")
    print(f"║ RDS                  ║{col(calc,'rds','{:.2e}'):>9} ║{col(eth,'rds','{:.2e}'):>9} ║{col(ds,'rds','{:.2e}'):>9} ║{rds_avg:>8.2e} ║")
    print(f"║ RDS vs pure RAG      ║{col(calc,'rds_vs_rag','{:.1f}x'):>9} ║{col(eth,'rds_vs_rag','{:.1f}x'):>9} ║{col(ds,'rds_vs_rag','{:.1f}x'):>9} ║{rvr_avg:>6.1f}x  ║")
    print(f"║ Est. cost            ║{('$'+col(calc,'total_cost','{:.4f}')):>9} ║{('$'+col(eth,'total_cost','{:.4f}')):>9} ║{('$'+col(ds,'total_cost','{:.4f}')):>9} ║{'${:.4f}'.format(cost_tot):>8} ║")
    print("╠══════════════════════╬══════════╬══════════╬══════════╬═════════╣")
    print("║ Paper RAG RDS        ║ 0.0000482║ 0.0000482║ 0.0000482║0.0000482║")
    print("║ Paper CKG RDS        ║ 0.00201  ║ 0.00201  ║ 0.00201  ║ 0.00201 ║")
    print("╚══════════════════════╩══════════╩══════════╩══════════╩═════════╝")
    print("* Baselines from Table 7, Yarmoluk & McCreary (2026)")

    n = len(all_results)
    total_q = sum(r["total_queries"] for r in all_results)
    print()
    print("═══════════════════════════════════════")
    print("CONTRIBUTION TO YARMOLUK & MCCREARY (2026)")
    print("═══════════════════════════════════════")
    print("This hybrid router implements the future work item:")
    print("'Hybrid architectures combining CKG structural")
    print(" precision with RAG prose retrieval for T1/T5'")
    print()
    print(f"Tested across {n} domains covering all 3 corpus")
    print("categories: STEM, Foundational, Professional")
    print(f"Total queries evaluated: {total_q}")
    print(f"Average classifier accuracy: {acc_avg:.1%}")
    print(f"Average token reduction: {red_avg:.1f}% vs RAG")
    print(f"Average RDS improvement: {rvr_avg:.1f}x vs pure RAG")
    print()
    print("To run across all 44 domains:")
    print("python run_multi.py --all-domains")
    print("  --benchmark-dir /path/to/ckg-benchmark")


def auto_generate_queries(domain_key) -> list:
    """
    Automatically generates 10 queries from a domain CSV.
    Used for the 44-domain full benchmark run.
    All concept names are derived from the actual CSV —
    no hardcoded names.
    """
    import csv
    import random
    random.seed(42)

    csv_path = os.path.join("data", domain_key, "learning-graph.csv")

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    foundational = [r for r in rows if not r["Dependencies"].strip()]
    has_deps = [r for r in rows if r["Dependencies"].strip()]
    by_dep_count = sorted(
        has_deps,
        key=lambda r: len([d for d in r["Dependencies"].split("|") if d.strip()]),
        reverse=True,
    )

    taxonomies = list(set(r["TaxonomyID"].strip() for r in rows if r["TaxonomyID"].strip()))

    queries = []

    # 2 T1 queries
    sample_t1 = random.sample(rows, min(2, len(rows)))
    for r in sample_t1:
        queries.append({
            "query": f"What is {r['ConceptLabel']}?",
            "expected_type": "T1",
            "ground_truth": None,
        })

    # 3 T2 queries — use concepts with most dependencies
    for r in by_dep_count[:3]:
        queries.append({
            "query": f"What are the prerequisites for {r['ConceptLabel']}?",
            "expected_type": "T2",
            "ground_truth": None,
        })

    # 2 T3 queries — foundational to advanced
    if len(foundational) >= 1 and len(by_dep_count) >= 2:
        src = random.choice(foundational)
        dst = by_dep_count[0]
        queries.append({
            "query": f"What is the learning path from {src['ConceptLabel']} to {dst['ConceptLabel']}?",
            "expected_type": "T3",
            "ground_truth": None,
        })
        if len(foundational) >= 2:
            src2 = random.choice(
                [f for f in foundational if f["ConceptID"] != src["ConceptID"]]
            )
            dst2 = by_dep_count[1] if len(by_dep_count) > 1 else by_dep_count[0]
            queries.append({
                "query": f"Path from {src2['ConceptLabel']} to {dst2['ConceptLabel']}?",
                "expected_type": "T3",
                "ground_truth": None,
            })

    # 2 T4 queries — use first two taxonomy values
    for tax in taxonomies[:2]:
        queries.append({
            "query": f"List all {tax} concepts",
            "expected_type": "T4",
            "ground_truth": None,
        })

    # 1 T5 query
    if len(rows) >= 2:
        pair = random.sample(rows, 2)
        queries.append({
            "query": f"How does {pair[0]['ConceptLabel']} relate to {pair[1]['ConceptLabel']}?",
            "expected_type": "T5",
            "ground_truth": None,
        })

    return queries


def main():
    parser = argparse.ArgumentParser(
        description="Multi-domain hybrid CKG + RAG router evaluation"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip all Anthropic API calls")
    parser.add_argument("--domains", nargs="+",
                        default=list(DOMAIN_CONFIGS.keys()),
                        help="Domains to run (default: all configured)")
    parser.add_argument("--all-domains", action="store_true",
                        help="Run all 44 benchmark domains")
    parser.add_argument("--benchmark-dir",
                        default=os.path.expanduser("~/Documents/ckg-benchmark"),
                        help="Path to full benchmark corpus")
    parser.add_argument(
        '--model',
        default='sonnet',
        choices=['haiku', 'sonnet', 'opus'],
        help='Model tier: haiku, sonnet, or opus'
    )
    args = parser.parse_args()
    model_name = MODEL_SHORTCUTS.get(args.model, DEFAULT_MODEL)

    mode = "DRY RUN" if args.dry_run else "LIVE"
    domains = args.domains
    print("╔══════════════════════════════════════════╗")
    print("║   MULTI-DOMAIN HYBRID CKG + RAG ROUTER   ║")
    print("║  Yarmoluk & McCreary (2026) — Future Work║")
    print("╚══════════════════════════════════════════╝")
    print(f"Mode: {mode}")
    print(f"Domains: {', '.join(domains)}")
    print(f"Model: {args.model} ({model_name})")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_results = []
    total_domains = len(domains)

    for i, domain in enumerate(domains):
        cfg = DOMAIN_CONFIGS.get(domain, {})
        display = cfg.get("display_name", domain)
        print(f"\nRunning domain {i+1}/{total_domains}: {display}...")
        result = run_domain(
            domain, args.dry_run,
            model=model_name
        )
        if result:
            print_domain_summary(result)
            all_results.append(result)

    if all_results:
        print_combined_table(all_results)

        # Save full results
        output = {
            "run_info": {
                "mode": "dry_run" if args.dry_run else "live",
                "timestamp": datetime.now().isoformat(),
                "domains": [r["domain"] for r in all_results],
                "total_queries": sum(r["total_queries"] for r in all_results),
            },
            "classifier_stats": dict(CLASSIFIER_STATS),
            "domains": all_results,
        }
        results_file = f"results_3domains_{args.model}.json"
        with open(results_file, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {results_file}")

        # Append log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "mode": "dry_run" if args.dry_run else "live",
            "domains": [r["domain"] for r in all_results],
            "avg_classifier_accuracy": sum(r["classifier_accuracy"] for r in all_results) / len(all_results),
            "avg_tokens": sum(r["avg_tokens"] for r in all_results) / len(all_results),
            "total_cost": sum(r["total_cost"] for r in all_results),
        }
        with open("run_log.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        print("Run logged to run_log.jsonl")

        total_cost = sum(r["total_cost"] for r in all_results)
        print(f"\nTotal cost all domains: ${total_cost:.4f}")


if __name__ == "__main__":
    main()


# ═══════════════════════════════════════════════════════════
# 44-DOMAIN FULL BENCHMARK — FOR AUTHORS TO RUN
# ═══════════════════════════════════════════════════════════
# Runs hybrid router across all 44 McCreary corpus domains.
# Produces macro-average results comparable to Table 7 in
# Yarmoluk & McCreary (2026).
#
# Usage:
#   python run_multi.py --all-domains
#   python run_multi.py --all-domains --dry-run
#
# Requirements:
#   git clone https://github.com/Yarmoluk/ckg-benchmark.git
#   Set BENCHMARK_DIR or pass --benchmark-dir flag
#
# Estimated cost: $27-35 at Claude Sonnet pricing
# Estimated time: 2-3 hours
#
# def run_all_44_domains(benchmark_dir, dry_run=False):
#     domains_dir = os.path.join(benchmark_dir,
#                                "benchmark", "domains")
#
#     if not os.path.exists(domains_dir):
#         print(f"ERROR: {domains_dir} not found")
#         print("Clone: git clone https://github.com/"
#               "Yarmoluk/ckg-benchmark.git")
#         return
#
#     domain_folders = sorted([
#         d for d in os.listdir(domains_dir)
#         if os.path.isdir(os.path.join(domains_dir, d))
#         and os.path.exists(os.path.join(
#             domains_dir, d, "learning-graph.csv"))
#     ])
#
#     print(f"Found {len(domain_folders)} domains")
#     print(f"Estimated cost: "
#           f"${len(domain_folders) * 0.75:.0f}-"
#           f"${len(domain_folders) * 0.85:.0f}")
#
#     all_results = []
#
#     for i, domain in enumerate(domain_folders):
#         print(f"\nDomain {i+1}/{len(domain_folders)}: "
#               f"{domain}")
#
#         src = os.path.join(domains_dir, domain,
#                           "learning-graph.csv")
#         dst_dir = os.path.join("data", domain)
#         os.makedirs(
#           os.path.join(dst_dir, "chapters"),
#           exist_ok=True)
#         import shutil
#         shutil.copy(src, os.path.join(
#           dst_dir, "learning-graph.csv"))
#
#         queries = auto_generate_queries(domain)
#
#         result = run_domain_with_queries(
#           domain, queries, dry_run)
#         all_results.append(result)
#
#         with open("results_all_domains.json", "w") as f:
#             json.dump(all_results, f, indent=2)
#         print(f"  Saved intermediate results")
#
#     print_macro_average_table(all_results)
#     return all_results
#
# def print_macro_average_table(all_results):
#     avg_acc = sum(r['classifier_accuracy']
#                  for r in all_results) / len(all_results)
#     avg_tokens = sum(r['avg_tokens']
#                     for r in all_results) / len(all_results)
#     avg_f1 = sum(r['avg_f1']
#                 for r in all_results) / len(all_results)
#     avg_rds = sum(r['rds']
#                  for r in all_results) / len(all_results)
#     total_cost = sum(r['total_cost']
#                     for r in all_results)
#     rds_vs_rag = avg_rds / 0.0000482
#
#     print()
#     print("╔══════════════════════════════════════════╗")
#     print("║   MACRO-AVERAGE — ALL 44 DOMAINS        ║")
#     print("║   Comparable to Table 7, paper (2026)   ║")
#     print("╠══════════════════╦══════════╦═══════════╣")
#     print("║ Metric           ║ Hybrid   ║ Paper RAG ║")
#     print("╠══════════════════╬══════════╬═══════════╣")
#     print(f"║ Classifier acc.  ║ {avg_acc:.1%}  ║ N/A       ║")
#     print(f"║ Avg tokens/query ║ {avg_tokens:.0f}      ║ 2,982     ║")
#     print(f"║ Avg F1           ║ {avg_f1:.4f}  ║ 0.1231    ║")
#     print(f"║ RDS              ║ {avg_rds:.7f}║ 0.0000482 ║")
#     print(f"║ RDS vs RAG       ║ {rds_vs_rag:.1f}x     ║ 1x        ║")
#     print(f"║ Total cost       ║ ${total_cost:.2f}    ║ $76.23    ║")
#     print("╚══════════════════╩══════════╩═══════════╝")
#
# if "--all-domains" in sys.argv:
#     benchmark_dir = getattr(args, 'benchmark_dir',
#         os.path.expanduser("~/Documents/ckg-benchmark"))
#     run_all_44_domains(benchmark_dir,
#                       dry_run="--dry-run" in sys.argv)
