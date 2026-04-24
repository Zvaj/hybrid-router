DATA_SCIENCE_QUERIES = [
    # T1 — explanatory, RAG routed
    {
        "query": "What is machine learning?",
        "expected_type": "T1",
        "ground_truth": None,
    },
    {
        "query": "Explain what gradient descent is and how it works",
        "expected_type": "T1",
        "ground_truth": None,
    },
    {
        "query": "How does a neural network learn?",
        "expected_type": "T1",
        "ground_truth": None,
    },
    # T2 — prerequisites, CKG routed
    {
        "query": "What are the prerequisites for Linear Regression?",
        "expected_type": "T2",
        "ground_truth": [
            "Data",
            "Variables",
            "Descriptive Statistics",
            "Dependent Variable",
            "Independent Variable",
            "Correlation",
            "Regression Analysis",
        ],
    },
    {
        "query": "What do I need to know before studying Neural Networks?",
        "expected_type": "T2",
        "ground_truth": [
            "Regression Analysis",
            "Linear Regression",
            "Data Science",
            "Machine Learning",
        ],
    },
    {
        "query": "Prerequisites for understanding Train Test Split?",
        "expected_type": "T2",
        "ground_truth": [
            "Regression Analysis",
            "Package Management",
            "Import Statement",
            "Prediction",
            "Data",
            "Linear Regression",
            "Python Libraries",
            "Model Performance",
            "Dataset",
            "Scikit-learn Library",
            "Testing Data",
            "Training Data",
        ],
    },
    {
        "query": "What should I learn before Cross-Validation?",
        "expected_type": "T2",
        "ground_truth": [
            "Prediction",
            "Linear Regression",
            "Python Libraries",
            "Dataset",
            "Model Complexity",
            "Model Performance",
            "Scikit-learn Library",
            "Testing Data",
            "Training Data",
            "Overfitting",
            "Train Test Split",
        ],
    },
    # T3 — learning path, CKG routed
    {
        "query": "What is the learning path from Python Programming to Linear Regression?",
        "expected_type": "T3",
        "ground_truth": [
            "Python Programming",
            "Variables",
            "Independent Variable",
            "Regression Analysis",
            "Linear Regression",
        ],
    },
    {
        "query": "Path from Data to Train Test Split?",
        "expected_type": "T3",
        "ground_truth": [
            "Data",
            "Dataset",
            "Training Data",
            "Train Test Split",
        ],
    },
    # T4 — category listing, CKG routed
    {
        "query": "List all foundational data science concepts",
        "expected_type": "T4",
        "ground_truth": [
            "Data Science",
            "Python Programming",
            "Data",
            "Variables",
            "Data Types",
            "Numerical Data",
            "Categorical Data",
        ],
    },
    {
        "query": "Show me all the machine learning concepts",
        "expected_type": "T4",
        "ground_truth": [
            "Machine Learning",
            "Supervised Learning",
            "Unsupervised Learning",
            "Classification",
            "Clustering",
        ],
    },
    # T5 — relationship/comparison, RAG routed
    {
        "query": "How does overfitting relate to the bias-variance tradeoff?",
        "expected_type": "T5",
        "ground_truth": None,
    },
    {
        "query": "What is the connection between normal distribution and confidence intervals?",
        "expected_type": "T5",
        "ground_truth": None,
    },
    # Ambiguous — classifier edge cases
    {
        "query": "What comes after learning linear regression?",
        "expected_type": "T2",
        "ground_truth": None,
    },
    {
        "query": "What are all the statistics concepts?",
        "expected_type": "T4",
        "ground_truth": None,
    },
]
