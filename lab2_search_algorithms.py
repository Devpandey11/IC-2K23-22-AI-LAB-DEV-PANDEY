"""Lab 2: BFS, DFS, Hill Climbing, Greedy Best First Search, A*."""
from collections import deque
import heapq

GRAPH = {
    "A": {"B": 1, "C": 4}, "B": {"D": 2, "E": 5},
    "C": {"F": 3}, "D": {}, "E": {"G": 2}, "F": {"G": 1}, "G": {}
}
H = {"A": 6, "B": 4, "C": 3, "D": 4, "E": 2, "F": 1, "G": 0}

def bfs(start, goal):
    q, seen = deque([(start, [start])]), {start}
    while q:
        node, path = q.popleft()
        if node == goal: return path
        for nxt in GRAPH[node]:
            if nxt not in seen:
                seen.add(nxt); q.append((nxt, path + [nxt]))
    return None

def dfs(start, goal):
    stack, seen = [(start, [start])], set()
    while stack:
        node, path = stack.pop()
        if node == goal: return path
        if node not in seen:
            seen.add(node)
            for nxt in reversed(list(GRAPH[node])):
                stack.append((nxt, path + [nxt]))
    return None

def hill_climbing(start, goal):
    node, path = start, [start]
    while node != goal:
        choices = GRAPH[node]
        if not choices: return None
        nxt = min(choices, key=lambda x: H[x])
        if H[nxt] >= H[node]: return None  # local optimum / no improvement
        node = nxt; path.append(node)
    return path

def greedy_best_first(start, goal):
    heap, seen = [(H[start], start, [start])], set()
    while heap:
        _, node, path = heapq.heappop(heap)
        if node == goal: return path
        if node in seen: continue
        seen.add(node)
        for nxt in GRAPH[node]:
            heapq.heappush(heap, (H[nxt], nxt, path + [nxt]))
    return None

def astar(start, goal):
    heap, best = [(H[start], 0, start, [start])], {start: 0}
    while heap:
        _, cost, node, path = heapq.heappop(heap)
        if node == goal: return path, cost
        if cost != best.get(node): continue
        for nxt, weight in GRAPH[node].items():
            new_cost = cost + weight
            if new_cost < best.get(nxt, float("inf")):
                best[nxt] = new_cost
                heapq.heappush(heap, (new_cost + H[nxt], new_cost, nxt, path + [nxt]))
    return None, float("inf")

if __name__ == "__main__":
    for name, fn in [("BFS", bfs), ("DFS", dfs), ("Hill Climbing", hill_climbing),
                     ("Best First Search", greedy_best_first)]:
        print(name, ":", fn("A", "G"))
    print("A*:", astar("A", "G"))
