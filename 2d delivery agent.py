#!/usr/bin/env python3
import argparse
import heapq
import time
import random
import sys
from collections import deque, defaultdict, namedtuple

Point = tuple[int, int]
Result = namedtuple("Result", ["path", "cost", "nodes_expanded", "time_secs"])

def neighbors_4(r, c, rows, cols):
    for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
        nr, nc = r+dr, c+dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield (nr, nc)

class Grid:
    def __init__(self, grid_costs, start, goal, static_blocked=None, dynamic_schedule=None):
        self.grid = grid_costs
        self.rows = len(grid_costs)
        self.cols = len(grid_costs[0]) if self.rows>0 else 0
        self.start = start
        self.goal = goal
        self.static_blocked = static_blocked if static_blocked is not None else set()
        self.dynamic_schedule = dynamic_schedule if dynamic_schedule is not None else defaultdict(set)

    def in_bounds(self, p: Point):
        r,c = p
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_blocked(self, p: Point, t: int = 0):
        if p in self.static_blocked:
            return True
        if t in self.dynamic_schedule and p in self.dynamic_schedule[t]:
            return True
        return False

    def cost(self, p: Point):
        r,c = p
        return self.grid[r][c]

    def pretty_print_with_path(self, path):
        grid_copy = [['#' if (r,c) in self.static_blocked else str(self.grid[r][c]) for c in range(self.cols)] for r in range(self.rows)]
        for (r,c) in path:
            grid_copy[r][c] = '*'
        sr,sc = self.start
        gr,gc = self.goal
        grid_copy[sr][sc] = 'S'
        grid_copy[gr][gc] = 'G'
        return "\n".join(" ".join(row) for row in grid_copy)

def parse_map_file(path: str):
    with open(path, 'r') as f:
        lines = [ln.rstrip() for ln in f.readlines()]

    if "" in lines:
        idx = lines.index("")
        grid_lines = lines[:idx]
        dyn_lines = lines[idx+1:]
    else:
        grid_lines = lines
        dyn_lines = []

    grid = []
    start = None
    goal = None
    static_blocked = set()

    for r, ln in enumerate(grid_lines):
        if ln.strip() == "":
            continue
        tokens = ln.split()
        row = []
        for c, tok in enumerate(tokens):
            if tok == '#':
                row.append(1)  
                static_blocked.add((r,c))
            elif tok.upper() == 'S':
                start = (r,c)
                row.append(1)
            elif tok.upper() == 'G':
                goal = (r,c)
                row.append(1)
            else:
                try:
                    val = int(tok)
                    if val < 1:
                        val = 1
                    row.append(val)
                except:
                    # fallback
                    row.append(1)
        grid.append(row)

    dynamic_schedule = defaultdict(set)
    for ln in dyn_lines:
        if ln.strip() == "":
            continue
        parts = ln.split()
        if len(parts) >= 3:
            t = int(parts[0])
            r = int(parts[1])
            c = int(parts[2])
            dynamic_schedule[t].add((r,c))

    if start is None or goal is None:
        raise ValueError("Map must contain S(start) and G(goal) tokens.")

    return Grid(grid, start, goal, static_blocked, dynamic_schedule)

def reconstruct_path(came_from, end):
    path = []
    cur = end
    while cur in came_from:
        path.append(cur)
        cur = came_from[cur]
    path.append(cur)
    path.reverse()
    return path

def bfs(grid: Grid, timeout=None):
    start_time = time.time()
    start = grid.start
    goal = grid.goal
    q = deque([start])
    came_from = {}
    visited = set([start])
    nodes = 0

    depth = {start: 0}

    while q:
        if timeout and (time.time() - start_time) > timeout:
            return Result(None, None, nodes, time.time() - start_time)
        current = q.popleft()
        nodes += 1
        t = depth[current]
        if grid.is_blocked(current, t):
            continue
        if current == goal:
            path = reconstruct_path(came_from, current)
            cost = sum(grid.cost(p) for p in path)
            return Result(path, cost, nodes, time.time() - start_time)
        for nb in neighbors_4(current[0], current[1], grid.rows, grid.cols):
            if nb in visited:
                continue
            if grid.is_blocked(nb, t+1):
                continue
            visited.add(nb)
            came_from[nb] = current
            depth[nb] = t+1
            q.append(nb)
    return Result(None, None, nodes, time.time() - start_time)

def ucs(grid: Grid, timeout=None):
    start_time = time.time()
    start = grid.start
    goal = grid.goal
    frontier = []
    heapq.heappush(frontier, (0, start, 0)) 
    came_from = {}
    cost_so_far = {start: 0}
    nodes = 0
    visited_time = dict()

    while frontier:
        if timeout and (time.time() - start_time) > timeout:
            return Result(None, None, nodes, time.time() - start_time)
        cur_cost, cur, cur_time = heapq.heappop(frontier)
        nodes += 1
        if grid.is_blocked(cur, cur_time):
            continue
        if cur == goal:
            path = reconstruct_path(came_from, cur)
            return Result(path, cur_cost, nodes, time.time() - start_time)
        for nb in neighbors_4(cur[0], cur[1], grid.rows, grid.cols):
            next_time = cur_time + 1
            if grid.is_blocked(nb, next_time):
                continue
            new_cost = cur_cost + grid.cost(nb)
            if nb not in cost_so_far or new_cost < cost_so_far[nb]:
                cost_so_far[nb] = new_cost
                came_from[nb] = cur
                heapq.heappush(frontier, (new_cost, nb, next_time))
    return Result(None, None, nodes, time.time() - start_time)

def manhattan(a: Point, b: Point):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def astar(grid: Grid, timeout=None):
    start_time = time.time()
    start = grid.start
    goal = grid.goal
    frontier = []
    start_h = manhattan(start, goal)
    heapq.heappush(frontier, (start_h + 0, 0, start, 0))
    cost_so_far = {start: 0}
    nodes = 0

    while frontier:
        if timeout and (time.time() - start_time) > timeout:
            return Result(None, None, nodes, time.time() - start_time)
        priority, cur_cost, cur, cur_time = heapq.heappop(frontier)
        nodes += 1
        if grid.is_blocked(cur, cur_time):
            continue
        if cur == goal:
            path = reconstruct_path(came_from, cur)
            return Result(path, cur_cost, nodes, time.time() - start_time)
        for nb in neighbors_4(cur[0], cur[1], grid.rows, grid.cols):
            next_time = cur_time + 1
            if grid.is_blocked(nb, next_time):
                continue
            new_cost = cur_cost + grid.cost(nb)
            if nb not in cost_so_far or new_cost < cost_so_far[nb]:
                cost_so_far[nb] = new_cost
                came_from[nb] = cur
                priority = new_cost + manhattan(nb, goal)
                heapq.heappush(frontier, (priority, new_cost, nb, next_time))
    return Result(None, None, nodes, time.time() - start_time)

def path_cost(grid: Grid, path):
    return sum(grid.cost(p) for p in path)

def valid_path(grid: Grid, path):
    for i in range(len(path)-1):
        a = path[i]; b = path[i+1]
        if b not in list(neighbors_4(a[0], a[1], grid.rows, grid.cols)):
            return False
        if grid.is_blocked(b, i+1):
            return False
    return True

def generate_random_path(grid: Grid, max_len=200):
    cur = grid.start
    path = [cur]
    attempts = 0
    while cur != grid.goal and len(path) < max_len and attempts < max_len*5:
        attempts += 1
        nbrs = list(neighbors_4(cur[0], cur[1], grid.rows, grid.cols))
        nbrs.sort(key=lambda p: manhattan(p, grid.goal))
        if random.random() < 0.8:
            choice = nbrs[0]
        else:
            choice = random.choice(nbrs)
        if choice in path:  
            choice = random.choice(nbrs)
            if choice in path:
                break
        path.append(choice)
        cur = choice
    if path[-1] != grid.goal:
        return None
    return path

def mutate_path(grid: Grid, path):
    if len(path) < 4:
        return path[:]
    i = random.randint(1, len(path)-3)
    j = random.randint(i+1, min(len(path)-2, i+5))
    a = path[i-1]
    b = path[j+1]
    cur = a
    new_sub = [a]
    steps = 0
    while cur != b and steps < 30:
        nbrs = list(neighbors_4(cur[0], cur[1], grid.rows, grid.cols))
        nbrs.sort(key=lambda p: manhattan(p, b))
        cur = nbrs[0] if random.random() < 0.7 else random.choice(nbrs)
        new_sub.append(cur)
        steps += 1
    if new_sub[-1] != b:
        return path[:]
    new_path = path[:i] + new_sub[1:-1] + path[j+1:]
    seen = set()
    pruned = []
    for p in new_path:
        if p in seen:
            continue
        pruned.append(p)
        seen.add(p)
    return pruned

def local_search_replan(grid: Grid, max_restarts=20, max_iters=200, timeout=None):
    start_time = time.time()
    best_path = None
    best_cost = float('inf')
    nodes = 0
    for restart in range(max_restarts):
        if timeout and (time.time() - start_time) > timeout:
            break
        seed_res = astar(grid, timeout=timeout)
        seed = seed_res.path if seed_res and seed_res.path else generate_random_path(grid)
        if not seed:
            continue
        current = seed
        current_cost = path_cost(grid, current)
        if not valid_path(grid, current):
            trials = 0
            while trials < 10 and not valid_path(grid, current):
                current = mutate_path(grid, current)
                trials += 1
        for it in range(max_iters):
            if timeout and (time.time() - start_time) > timeout:
                break
            neighbor = mutate_path(grid, current)
            nodes += 1
            if not neighbor:
                continue
            if not valid_path(grid, neighbor):
                continue
            ncost = path_cost(grid, neighbor)
            if ncost < current_cost:
                current = neighbor
                current_cost = ncost
            # keep track of best found
            if current_cost < best_cost and valid_path(grid, current):
                best_cost = current_cost
                best_path = current[:]
            # small random hill climb acceptance
            if random.random() < 0.01:
                current = neighbor
                current_cost = ncost
    return Result(best_path, best_cost if best_path else None, nodes, time.time()-start_time)

def run_planner(grid: Grid, algo: str, timeout=None):
    algo = algo.lower()
    if algo == 'bfs':
        return bfs(grid, timeout=timeout)
    if algo == 'ucs':
        return ucs(grid, timeout=timeout)
    if algo == 'astar':
        return astar(grid, timeout=timeout)
    if algo == 'local':
        return local_search_replan(grid, timeout=timeout)
    raise ValueError("Unknown algorithm")

def write_example_maps():
    import os
    os.makedirs("examples", exist_ok=True)
    small = """1 1 1 G
1 # 1 1
S 1 1 1
"""
    medium = """1 1 1 1 1 1
1 5 5 1 1 1
1 # 1 1 2 G
1 1 1 5 1 1
S 1 1 1 1 1
"""
    large = """1 1 1 1 1 1 1 1 1 1
1 5 5 5 1 2 2 1 1 1
1 # # 1 1 2 9 9 1 1
1 1 1 1 # 1 1 1 1 1
S 1 1 2 1 1 1 1 1 G
"""
    dynamic = """1 1 1 1 1
1 5 1 5 1
1 # 1 # 1
S 1 1 1 G

# dynamic schedule: t r c
1 1 2
2 2 2
3 2 1
3 0 2
"""
    with open("examples/small.txt","w") as f: f.write(small)
    with open("examples/medium.txt","w") as f: f.write(medium)
    with open("examples/large.txt","w") as f: f.write(large)
    with open("examples/dynamic.txt","w") as f: f.write(dynamic)
    print("Wrote example maps to ./examples/*.txt")

def main():
    parser = argparse.ArgumentParser(description="Autonomous delivery agent planners (BFS, UCS, A*) with local replanning.")
    parser.add_argument('--algo', choices=['bfs','ucs','astar','local'], default='astar', help='Algorithm to run')
    parser.add_argument('--map', required=False, help='Map file path (see README in header). If omitted, example maps will be written and used.')
    parser.add_argument('--dynamic', action='store_true', help='Enable dynamic schedule (if present in map).')
    parser.add_argument('--timeout', type=float, default=10.0, help='Timeout in seconds per planner run.')
    parser.add_argument('--demo', action='store_true', help='Run all planners on all example maps and print results.')
    args = parser.parse_args()

    if not args.map and not args.demo:
        write_example_maps()
        print("No map specified; created example maps in ./examples. Re-run with --map examples/dynamic.txt etc.")
        return

    if args.demo:
        write_example_maps()
        maps = ["examples/small.txt","examples/medium.txt","examples/large.txt","examples/dynamic.txt"]
        for mpath in maps:
            print("\n=== Running on map:", mpath)
            grid = parse_map_file(mpath)
            print("Grid size:", grid.rows, "x", grid.cols)
            for algo in ['bfs','ucs','astar','local']:
                print(f"\n-- {algo.upper()} --")
                res = run_planner(grid, algo, timeout=args.timeout)
                if res.path:
                    print("Found path length:", len(res.path), "cost:", res.cost, "nodes:", res.nodes_expanded, "time(s):", f"{res.time_secs:.4f}")
                    print("Path:", res.path)
                else:
                    print("No path found (or timed out). nodes:", res.nodes_expanded, "time(s):", f"{res.time_secs:.4f}")
        return

    grid = parse_map_file(args.map)
    print(f"Loaded map {args.map} size {grid.rows}x{grid.cols}. Start={grid.start} Goal={grid.goal}. Dynamic entries: {sum(len(v) for v in grid.dynamic_schedule.values())}")
    print("Running", args.algo.upper(), "...")
    res = run_planner(grid, args.algo, timeout=args.timeout if args.dynamic else args.timeout)
    if res.path:
        print("Found path! length:", len(res.path), "cost:", res.cost, "nodes:", res.nodes_expanded, "time(s):", f"{res.time_secs:.4f}")
        print(grid.pretty_print_with_path(res.path))
    else:
        print("No path found or timed out. nodes:", res.nodes_expanded, "time(s):", f"{res.time_secs:.4f}")

if __name__ == '__main__':
    main()
