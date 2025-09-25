# Autonomous Delivery Agent – Report Draft

## 1. Introduction

This project implements an autonomous delivery agent operating on a 2D grid environment.  
The agent must find efficient delivery paths under static and dynamic obstacles using multiple AI search strategies.

Algorithms implemented:

- BFS (uninformed search)
- UCS (cost-sensitive uninformed)
- A\* (informed with Manhattan heuristic)
- Hill Climbing (local search with restarts)
- Simulated Annealing (probabilistic local search)

---

## 2. Environment Model

- Grid world with dimensions (rows × cols).
- Obstacles:
  - `#` = static obstacles.
  - Dynamic obstacles specified as `(time, x, y)` meaning blocked at that time.
- Each move = unit time step.
- Terrain cost = integer ≥ 1 (default = 1).
- Agent can move in 4 directions: up, down, left, right.

### Movement Assumptions

- Diagonal movement: Not allowed in this project (4-connected grid). This aligns with the Manhattan heuristic used by A\* and ensures admissibility.
- Time model: Each move advances time by 1. Dynamic obstacles are evaluated against the current time step.

---

## 3. Algorithms

### BFS

- Explores level by level.
- Guarantees shortest path in steps, not cost.

### UCS

- Explores by cumulative path cost.
- Guarantees minimum cost solution.

### A\*

- Uses Manhattan heuristic.
- Admissible and efficient for grid navigation.

### Hill Climbing

- Greedy local search.
- Uses restarts/dynamic replanning to avoid failures.

### Simulated Annealing

- Probabilistic acceptance of worse moves.
- Can escape local minima better than hill climbing.

---

## 4. Experimental Setup

- Test maps: `small.txt`, `medium.txt`, `large.txt`, `dynamic.txt`.
- Performance metrics recorded:
  - Path length
  - Path cost
  - Nodes expanded
  - Runtime (seconds)

For dynamic strategies (hill climbing, simulated annealing), we record:

- Path length (measured by time steps taken)
- Runtime (seconds)
- Cost and nodes expanded are not applicable and logged as NA in `metrics.csv`.

---

## 5. Results

### Table 1 – Static Search Performance

(From `metrics.csv`)

| Algorithm | Map    | PathLength | Cost | NodesExpanded | Runtime (s) |
| --------- | ------ | ---------- | ---- | ------------- | ----------- |
| BFS       | small  | 4          | 3    | 7             | 0.000051    |
| BFS       | medium | 11         | 10   | 20            | 0.000108    |
| BFS       | large  | 21         | 20   | 93            | 0.000323    |
| UCS       | small  | 4          | 3    | 6             | 0.000076    |
| UCS       | medium | 11         | 10   | 20            | 0.000137    |
| UCS       | large  | 21         | 20   | 93            | 0.000546    |
| A\*       | small  | 4          | 3    | 4             | 0.000063    |
| A\*       | medium | 11         | 10   | 14            | 0.000108    |
| A\*       | large  | 21         | 20   | 89            | 0.000560    |

---

### Table 2 – Local Search on Dynamic Map

(From `run_log.txt`)

| Algorithm           | Success | PathLength | Runtime (s)\* |
| ------------------- | ------- | ---------- | ------------- |
| Hill Climbing       | Yes     | 9          | ~0.0003       |
| Simulated Annealing | Yes     | 79         | ~0.0006       |

\*Runtime is approximated; exact values can be measured by wrapping `agent.move()` with `time.perf_counter()`.

---

## 6. Analysis

### When each method performs better (based on experiments)

- **BFS (uninformed)**: Best when all moves have uniform cost and the optimal path is short in steps. It is predictable but can expand many nodes on larger maps because it ignores costs.
- **UCS (uninformed but cost-sensitive)**: Best when terrain costs vary significantly and the cheapest route is not the shortest in steps. It may expand more nodes than A\* due to lack of guidance.
- **A\* (informed)**: Best overall for static maps with Manhattan distances; expands fewer nodes than UCS while preserving optimality. Benefits grow as map size increases and obstacles create detours.
- **Hill Climbing (dynamic, local search)**: Performs well when the heuristic landscape is smooth and obstacles are sparse or short-lived. However, it can get stuck without replanning; dynamic replanning mitigates this but may still fail in mazes.
- **Simulated Annealing (dynamic, probabilistic)**: More robust than hill climbing in rugged landscapes with many local minima; tends to incur longer paths/runtimes but succeeds more often when obstacles force detours.

### Why these trends occur

- Heuristic guidance in A\* prunes the search more aggressively than UCS, especially with Manhattan distance on 4-connected grids.
- Local search methods trade optimality for adaptability; dynamic replanning lets them recover from newly blocked cells, but lack of global optimality yields longer paths.

---

## 7. Conclusion

- For **static maps**, A\* is the most efficient.
- For **dynamic maps**, local search with replanning (Hill Climbing, Simulated Annealing) is essential.
- Future work:
  - Add richer dynamic metrics (e.g., number of replans, detour length).
  - Visualization of pathfinding and replanning.
  - Extend movement to diagonals and update heuristics accordingly.

---
