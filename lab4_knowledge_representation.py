"""Lab 4: Knowledge representation, rule-based system, semantic network."""
# Facts are represented as triples: (subject, relation, object).
knowledge_base = [
    ("Tweety", "is_a", "Bird"),
    ("Bird", "has", "Wings"),
    ("Bird", "can", "Fly"),
    ("Penguin", "is_a", "Bird"),
    ("Penguin", "cannot", "Fly"),
    ("Pingu", "is_a", "Penguin"),
]

def query(subject, relation):
    return [obj for s, r, obj in knowledge_base if s == subject and r == relation]

print("Semantic network facts:")
for fact in knowledge_base:
    print(fact)

# Simple forward-chaining rule system.
facts = {"fever", "cough"}
rules = [
    ({"fever", "cough"}, "possible_flu"),
    ({"possible_flu"}, "suggest_rest"),
]
changed = True
while changed:
    changed = False
    for required, conclusion in rules:
        if required.issubset(facts) and conclusion not in facts:
            facts.add(conclusion)
            changed = True
            print("Rule fired:", required, "=>", conclusion)

print("\nInferred facts:", sorted(facts))
print("Tweety is:", query("Tweety", "is_a"))
print("Bird has:", query("Bird", "has"))
