"""Lab 1: Environment setup, a simple production system, Water Jug Problem."""
import sys
from collections import deque

def environment_check():
    print("Python:", sys.version.split()[0])
    for package in ("numpy", "pandas", "matplotlib", "sklearn", "torch"):
        try:
            module = __import__(package)
            print(f"{package}: installed", getattr(module, "__version__", ""))
        except ImportError:
            print(f"{package}: NOT installed")

# Production system: rules are applied to the current state until goal is reached.
def production_system():
    state = {"temperature": 38, "fan": "OFF"}
    rules = [
        (lambda s: s["temperature"] > 30 and s["fan"] == "OFF",
         lambda s: s.update(fan="ON")),
        (lambda s: s["temperature"] <= 30 and s["fan"] == "ON",
         lambda s: s.update(fan="OFF")),
    ]
    print("\nProduction system initial state:", state)
    for condition, action in rules:
        if condition(state):
            action(state)
            print("Rule fired ->", state)
    return state

# State is (amount in 4L jug, amount in 3L jug). Goal: 2L in 4L jug.
def water_jug():
    start, goal = (0, 0), 2
    q = deque([(start, [start])])
    visited = {start}
    while q:
        (a, b), path = q.popleft()
        if a == goal:
            print("\nWater Jug solution (4L, 3L):")
            for step, state in enumerate(path):
                print(step, state)
            return path
        # Fill, empty, and pour operations.
        na = [
            (4, b), (a, 3), (0, b), (a, 0),
            (a - min(a, 3-b), b + min(a, 3-b)),
            (a + min(b, 4-a), b - min(b, 4-a)),
        ]
        for nxt in na:
            if nxt not in visited:
                visited.add(nxt)
                q.append((nxt, path + [nxt]))
    print("No solution found.")
    return None

if __name__ == "__main__":
    environment_check()
    production_system()
    water_jug()
