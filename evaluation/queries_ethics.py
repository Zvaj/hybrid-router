ETHICS_QUERIES = [
    # T1 — explanatory, RAG routed
    {
        "query": "What is data-driven ethics?",
        "expected_type": "T1",
        "ground_truth": None,
    },
    {
        "query": "Explain what harm quantification means in ethical analysis",
        "expected_type": "T1",
        "ground_truth": None,
    },
    {
        "query": "How does systems thinking apply to ethical decision-making?",
        "expected_type": "T1",
        "ground_truth": None,
    },
    # T2 — prerequisites, CKG routed
    {
        "query": "What are the prerequisites for understanding DALYs?",
        "expected_type": "T2",
        "ground_truth": [
            "Data Literacy",
            "Quantitative Analysis",
            "Harm Definition",
            "Disability Weights",
            "Statistical Thinking",
            "Harm Quantification",
            "Years Lived with Disability",
            "Years of Life Lost",
            "Morbidity Rate",
            "Mortality Rate",
        ],
    },
    {
        "query": "What do I need to know before studying Causal Loop Diagrams?",
        "expected_type": "T2",
        "ground_truth": [
            "System Components",
            "Negative Feedback",
            "Positive Feedback",
            "Interconnections",
            "Balancing Loops",
            "Reinforcing Loops",
            "Feedback Loops",
        ],
    },
    {
        "query": "Prerequisites for understanding Intervention Hierarchy",
        "expected_type": "T2",
        "ground_truth": [
            "Complex Systems",
            "Critical Thinking",
            "Donella Meadows",
            "Systems Thinking",
            "High Leverage Points",
            "Low Leverage Points",
            "Leverage Points",
        ],
    },
    {
        "query": "What should I know before studying Corporate Accountability?",
        "expected_type": "T2",
        "ground_truth": [
            "Advocacy Strategies",
            "Social Cost Accounting",
            "Harm Quantification",
            "Policy Design",
            "Social Impact Assessment",
            "Ethics",
            "Regulatory Approaches",
            "Corporate Responsibility",
        ],
    },
    # T3 — learning path, CKG routed
    {
        "query": "What is the learning path from Research Methods to Source Triangulation?",
        "expected_type": "T3",
        "ground_truth": [
            "Research Methods",
            "Primary Sources",
            "Academic Sources",
            "Data Credibility",
            "Source Triangulation",
        ],
    },
    {
        "query": "Path from Ethics to Harm Quantification?",
        "expected_type": "T3",
        "ground_truth": [
            "Ethics",
            "Harm Definition",
            "Harm Quantification",
        ],
    },
    # T4 — category listing, CKG routed
    {
        "query": "List all foundational ethics concepts",
        "expected_type": "T4",
        "ground_truth": [
            "Ethics",
            "Data-Driven Ethics",
            "Traditional Moral Philosophy",
            "Ethical Reasoning",
            "Evidence-Based Ethics",
            "Scientific Method",
            "Critical Thinking",
        ],
    },
    {
        "query": "Show me all corporate ethics topics",
        "expected_type": "T4",
        "ground_truth": [
            "Corporate Responsibility",
            "ESG Metrics",
            "Sustainability Reporting",
            "Whistleblowing",
            "Corporate Accountability",
        ],
    },
    # T5 — relationship/comparison, RAG routed
    {
        "query": "How does confirmation bias relate to data credibility?",
        "expected_type": "T5",
        "ground_truth": None,
    },
    {
        "query": "What is the connection between harm quantification and advocacy?",
        "expected_type": "T5",
        "ground_truth": None,
    },
    # Ambiguous — classifier edge cases
    {
        "query": "What comes after understanding bias recognition?",
        "expected_type": "T2",
        "ground_truth": None,
    },
    {
        "query": "What are all the harm analysis concepts?",
        "expected_type": "T4",
        "ground_truth": None,
    },
]
