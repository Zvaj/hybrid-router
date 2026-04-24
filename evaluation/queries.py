TEST_QUERIES = [
    # T1 — definition/explanation, RAG routed
    {
        "query": "What is a derivative?",
        "expected_type": "T1",
        "ground_truth": None,
    },
    {
        "query": "Explain what a limit is in calculus",
        "expected_type": "T1",
        "ground_truth": None,
    },
    {
        "query": "How does the chain rule work?",
        "expected_type": "T1",
        "ground_truth": None,
    },
    {
        "query": "What is the fundamental theorem of calculus?",
        "expected_type": "T1",
        "ground_truth": None,
    },
    # T2 — prerequisites, CKG routed
    {
        "query": "What are the prerequisites for learning the chain rule?",
        "expected_type": "T2",
        "ground_truth": [
            "Composite Function",
            "Derivative Rules",
            "Function",
            "Function Notation",
            "Definition of Derivative",
        ],
    },
    {
        "query": "What do I need to know before studying implicit differentiation?",
        "expected_type": "T2",
        "ground_truth": [
            "Chain Rule",
            "Quotient Rule",
            "Derivative Rules",
            "Product Rule",
            "Definition of Derivative",
        ],
    },
    {
        "query": "Prerequisites for understanding Taylor Series?",
        "expected_type": "T2",
        "ground_truth": [
            "Integration by Parts",
            "Fundamental Theorem of Calculus",
            "Definite Integral",
        ],
    },
    # T3 — learning path, CKG routed
    {
        "query": "What is the learning path from Function to Chain Rule?",
        "expected_type": "T3",
        "ground_truth": [
            "Function",
            "Composite Function",
            "Derivative Rules",
            "Chain Rule",
        ],
    },
    {
        "query": "Path from Function to Taylor Series?",
        "expected_type": "T3",
        "ground_truth": [
            "Function",
            "Limit",
            "Definition of Derivative",
            "Derivative Rules",
            "Definite Integral",
            "Fundamental Theorem of Calculus",
            "Integration by Parts",
            "Taylor Series",
        ],
    },
    # T4 — category listing, CKG routed
    {
        "query": "List all foundational concepts in this course",
        "expected_type": "T4",
        "ground_truth": [
            "Function",
            "Domain and Range",
            "Function Notation",
            "Composite Function",
        ],
    },
    {
        "query": "Show me all the core calculus topics",
        "expected_type": "T4",
        "ground_truth": [
            "Limit",
            "Limit Laws",
            "Continuity",
            "Definition of Derivative",
            "Derivative Rules",
            "Chain Rule",
            "Product Rule",
            "Quotient Rule",
            "Definite Integral",
        ],
    },
    # T5 — relationship/comparison, RAG routed
    {
        "query": "How does the chain rule relate to composite functions?",
        "expected_type": "T5",
        "ground_truth": None,
    },
    {
        "query": "What is the connection between limits and derivatives?",
        "expected_type": "T5",
        "ground_truth": None,
    },
    # Ambiguous — classifier edge cases
    {
        "query": "What comes after understanding limits?",
        "expected_type": "T2",
        "ground_truth": None,
    },
    {
        "query": "I know functions, what should I learn next?",
        "expected_type": "T2",
        "ground_truth": None,
    },
    {
        "query": "Walk me through the concepts between Function and Chain Rule",
        "expected_type": "T3",
        "ground_truth": None,
    },
    {
        "query": "Give me all advanced topics",
        "expected_type": "T4",
        "ground_truth": [
            "Implicit Differentiation",
            "Related Rates",
            "Fundamental Theorem of Calculus",
            "Integration by Parts",
            "Taylor Series",
        ],
    },
    {
        "query": "Continuity",
        "expected_type": "T1",
        "ground_truth": None,
    },
]
