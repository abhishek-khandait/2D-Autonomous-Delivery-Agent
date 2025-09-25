from collections import deque
import heapq
import math
import random


# ---------- BFS ----------
def bfs(grid, start, goal):
    queue = deque([start])
    visited = set([start])
    parent = {start: None}
    nodes_expanded = 0

    while queue:
        x, y = queue.popleft()
        nodes_expanded += 1
        if (x, y) == goal:
            path = reconstruct_path(parent, goal)
            return path, len(path) - 1, nodes_expanded  # cost = steps

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if grid.is_valid(nx, ny, time=0) and (nx, ny) not in visited:
                visited.add((nx, ny))
                parent[(nx, ny)] = (x, y)
                queue.append((nx, ny))
    return None, float("inf"), nodes_expanded


# ---------- UCS ----------
def ucs(grid, start, goal):
    pq = [(0, start)]
    visited = set()
    parent = {start: None}
    cost_so_far = {start: 0}
    nodes_expanded = 0

    while pq:
        cost, (x, y) = heapq.heappop(pq)
        nodes_expanded += 1
        if (x, y) == goal:
            path = reconstruct_path(parent, goal)
            return path, cost, nodes_expanded

        if (x, y) in visited:
            continue
        visited.add((x, y))

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if grid.is_valid(nx, ny, time=0):
                new_cost = cost_so_far[(x, y)] + grid.get_cost(nx, ny)
                if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                    cost_so_far[(nx, ny)] = new_cost
                    parent[(nx, ny)] = (x, y)
                    heapq.heappush(pq, (new_cost, (nx, ny)))
    return None, float("inf"), nodes_expanded


# ---------- A* ----------
def a_star(grid, start, goal):
    pq = [(0, start)]
    cost_so_far = {start: 0}
    parent = {start: None}
    visited = set()
    nodes_expanded = 0

    while pq:
        f_score, (x, y) = heapq.heappop(pq)
        nodes_expanded += 1
        if (x, y) == goal:
            path = reconstruct_path(parent, goal)
            return path, cost_so_far[(x, y)], nodes_expanded

        if (x, y) in visited:
            continue
        visited.add((x, y))

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if grid.is_valid(nx, ny, time=0):
                new_cost = cost_so_far[(x, y)] + grid.get_cost(nx, ny)
                if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                    cost_so_far[(nx, ny)] = new_cost
                    parent[(nx, ny)] = (x, y)
                    h = abs(goal[0] - nx) + abs(goal[1] - ny)
                    f_score = new_cost + h
                    heapq.heappush(pq, (f_score, (nx, ny)))
    return None, float("inf"), nodes_expanded


# ---------- Local Search: Hill Climbing ----------
def hill_climbing(grid, start, goal, max_restarts=10):
    best_path = None
    for _ in range(max_restarts):
        current = start
        path = [current]
        visited = set([current])

        while current != goal:
            neighbors = [
                (current[0] + dx, current[1] + dy)
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]
                if grid.is_valid(current[0] + dx, current[1] + dy)
            ]
            neighbors = [n for n in neighbors if n not in visited]
            if not neighbors:
                break

            current = min(
                neighbors, key=lambda x: abs(x[0] - goal[0]) + abs(x[1] - goal[1])
            )
            path.append(current)
            visited.add(current)

        if path[-1] == goal:
            if best_path is None or len(path) < len(best_path):
                best_path = path
    return best_path


# ---------- Local Search: Simulated Annealing ----------
def simulated_annealing(
    grid, start, goal, max_iterations=500, temperature=100.0, cooling_rate=0.99
):
    current = start
    path = [current]

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    for _ in range(max_iterations):
        if current == goal:
            return path

        neighbors = [
            (current[0] + dx, current[1] + dy)
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]
            if grid.is_valid(current[0] + dx, current[1] + dy)
        ]

        if not neighbors:
            break

        next_node = random.choice(neighbors)
        delta_e = heuristic(current, goal) - heuristic(next_node, goal)

        if delta_e > 0 or math.exp(delta_e / temperature) > random.random():
            current = next_node
            path.append(current)

        temperature *= cooling_rate

    return path if path[-1] == goal else None


# ---------- Helper ----------
def reconstruct_path(parent, goal):
    path = []
    node = goal
    while node:
        path.append(node)
        node = parent[node]
    path.reverse()
    return path
