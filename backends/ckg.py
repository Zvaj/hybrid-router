import csv
from collections import deque


class CKGBackend:
    def __init__(self, csv_path):
        self.graph = {}
        self.label_index = {}
        self.reverse_index = {}

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = int(row["ConceptID"].strip())
                label = row["ConceptLabel"].strip()
                deps_raw = row["Dependencies"].strip()
                taxonomy = row["TaxonomyID"].strip()

                deps = [int(d.strip()) for d in deps_raw.split("|") if d.strip()] if deps_raw else []

                self.graph[cid] = {
                    "label": label,
                    "deps": deps,
                    "taxonomy": taxonomy,
                }
                self.label_index[label.lower()] = cid
                if cid not in self.reverse_index:
                    self.reverse_index[cid] = []

        for cid, data in self.graph.items():
            for dep_id in data["deps"]:
                if dep_id not in self.reverse_index:
                    self.reverse_index[dep_id] = []
                self.reverse_index[dep_id].append(cid)

        edge_count = sum(len(d["deps"]) for d in self.graph.values())
        print(f"CKG loaded: {len(self.graph)} concepts, {edge_count} edges")

    def find_concept(self, query):
        q = query.lower()

        # Stage 1: substring match
        # Prefer label-in-query (specific concept named in query) over query-in-label (broad)
        in_query = [(label, cid) for label, cid in self.label_index.items() if label in q]
        if in_query:
            in_query.sort(key=lambda x: len(x[0]), reverse=True)
            return in_query[0][1]
        q_in_label = [(label, cid) for label, cid in self.label_index.items() if q in label]
        if q_in_label:
            q_in_label.sort(key=lambda x: len(x[0]))
            return q_in_label[0][1]

        # Stage 2: word-level partial match (4+ char words)
        query_words = [w for w in q.split() if len(w) >= 4]
        if not query_words:
            return None
        best_cid, best_score = None, 0
        for label, cid in self.label_index.items():
            score = sum(1 for w in query_words if w in label)
            if score > best_score:
                best_score, best_cid = score, cid
        return best_cid if best_score > 0 else None

    def get_prerequisites(self, concept_id, hops=3):
        visited = set()
        queue = deque([(concept_id, 0)])
        ordered = []

        while queue:
            cid, depth = queue.popleft()
            if depth >= hops:
                continue
            for dep_id in self.graph.get(cid, {}).get("deps", []):
                if dep_id not in visited and dep_id in self.graph:
                    visited.add(dep_id)
                    ordered.append(dep_id)
                    queue.append((dep_id, depth + 1))

        ordered.reverse()
        return [self.graph[cid]["label"] for cid in ordered if cid in self.graph]

    def get_dependents(self, concept_id):
        return [
            self.graph[cid]["label"]
            for cid in self.reverse_index.get(concept_id, [])
            if cid in self.graph
        ]

    def get_path(self, from_id, to_id):
        if from_id == to_id:
            return [self.graph[from_id]["label"]] if from_id in self.graph else []

        visited = {from_id}
        queue = deque([[from_id]])

        while queue:
            path = queue.popleft()
            current = path[-1]
            neighbors = self.graph.get(current, {}).get("deps", []) + self.reverse_index.get(current, [])
            for neighbor in neighbors:
                if neighbor not in visited and neighbor in self.graph:
                    new_path = path + [neighbor]
                    if neighbor == to_id:
                        return [self.graph[cid]["label"] for cid in new_path]
                    visited.add(neighbor)
                    queue.append(new_path)

        return []

    # Maps user-facing category words to actual CSV taxonomy codes
    TAXONOMY_GROUPS = {
        "FOUND": ["FOUND"],
        "CORE":  ["LIMIT", "CONT", "DERIV", "DRULE", "CHAIN", "INTEG"],
        "ADV":   ["IMPL", "APPL", "ANAL", "OPT", "FTC", "TECH", "HIGH", "CURV", "ASYM", "RIEM"],
    }

    def get_category(self, group_key):
        codes = {c.upper() for c in self.TAXONOMY_GROUPS.get(group_key.upper(), [group_key.upper()])}
        return [
            data["label"]
            for data in self.graph.values()
            if data["taxonomy"].upper() in codes
        ]

    FORWARD_KEYWORDS = [
        "after", "next", "comes after", "learn next", "study next",
        "what's next", "move on from", "already know", "i know",
    ]

    def retrieve(self, query, query_type):
        if query_type == "T2":
            cid = self.find_concept(query)
            if cid is None:
                return None
            q = query.lower()
            if any(kw in q for kw in self.FORWARD_KEYWORDS):
                dependents = self.get_dependents(cid)
                return {
                    "concept": self.graph[cid]["label"],
                    "dependents": dependents,
                    "type": "dependents",
                }
            prereqs = self.get_prerequisites(cid)
            return {
                "concept": self.graph[cid]["label"],
                "prerequisites": prereqs,
                "type": "prerequisites",
            }

        elif query_type == "T3":
            found = []
            for label, cid in self.label_index.items():
                if label in query.lower():
                    found.append((len(label), cid, self.graph[cid]["label"]))
            found.sort(reverse=True)
            if len(found) < 2:
                return None
            _, from_id, from_label = found[0]
            _, to_id, to_label = found[1]
            path = self.get_path(from_id, to_id)
            return {
                "from_concept": from_label,
                "to_concept": to_label,
                "path": path,
                "type": "path",
            }

        elif query_type == "T4":
            q_up = query.upper()
            # Map user words to internal group keys
            if any(w in q_up for w in ["FOUND", "BASIC", "INTRODUCT"]):
                group = "FOUND"
            elif any(w in q_up for w in ["CORE", "MAIN", "PRIMARY", "CENTRAL", "STANDARD"]):
                group = "CORE"
            elif any(w in q_up for w in ["ADV", "HARD", "DIFFICULT", "COMPLEX", "HIGHER"]):
                group = "ADV"
            else:
                group = "CORE"  # sensible default when no category word detected
            concepts = self.get_category(group)
            return {
                "taxonomy": group,
                "concepts": concepts,
                "type": "category",
            }

        elif query_type == "T5":
            cid = self.find_concept(query)
            if cid is None:
                return None
            prereqs = self.get_prerequisites(cid, hops=2)
            dependents = self.get_dependents(cid)
            return {
                "concept": self.graph[cid]["label"],
                "related": prereqs + dependents,
                "type": "cross",
            }

        return None
