import os
import anthropic

CLASSIFIER_STATS = {
    "keyword_filter": 0,
    "llm_fallback": 0,
    "default": 0,
}

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

_TYPE_KEYWORDS = {
    "T1": T1_KEYWORDS,
    "T2": T2_KEYWORDS,
    "T3": T3_KEYWORDS,
    "T4": T4_KEYWORDS,
    "T5": T5_KEYWORDS,
}

_VALID_TYPES = {"T1", "T2", "T3", "T4", "T5"}


def classify_query(query, use_llm_fallback=True):
    q = query.lower()

    counts = {t: sum(1 for kw in kws if kw in q) for t, kws in _TYPE_KEYWORDS.items()}

    max_count = max(counts.values())
    if max_count >= 2:
        winner = max(counts, key=counts.get)
        CLASSIFIER_STATS["keyword_filter"] += 1
        return winner

    matched = [t for t, c in counts.items() if c == 1]
    if len(matched) == 1:
        CLASSIFIER_STATS["keyword_filter"] += 1
        return matched[0]

    if not use_llm_fallback:
        CLASSIFIER_STATS["default"] += 1
        return "T1"

    try:
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        prompt = (
            "Classify this query into exactly one type:\n"
            "T1=definition or explanation of a concept\n"
            "T2=prerequisites or what to learn before a concept\n"
            "T3=learning path or route between two concepts\n"
            "T4=list all concepts of a category type\n"
            "T5=relationship or comparison between two concepts\n"
            f"Query: {query}\n"
            "Reply with only the type code: T1, T2, T3, T4, or T5"
        )
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
        )
        result = message.content[0].text.strip().upper()
        if result not in _VALID_TYPES:
            CLASSIFIER_STATS["default"] += 1
            return "T1"
        CLASSIFIER_STATS["llm_fallback"] += 1
        return result
    except Exception:
        CLASSIFIER_STATS["default"] += 1
        return "T1"
