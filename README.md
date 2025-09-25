## Autonomous Delivery Agent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A small research/teaching project for grid-based pathfinding and dynamic replanning. It provides classic search algorithms (BFS, UCS, A\*), local search strategies (hill climbing, simulated annealing), a simple dynamic agent that replans when obstacles appear, and scripts to log and plot performance metrics.

### Features

- **Static pathfinding**: BFS, UCS, A\*
- **Local search / dynamic agent**: Hill Climbing, Simulated Annealing (with time-stepped replanning)
- **CLI** for running algorithms on text-based maps
- **Automatic logging** of runs and metrics
- **Plot generation** of runtime, nodes expanded, and path cost

## Project structure

```
maps/
  small.txt | medium.txt | large.txt | dynamic.txt
results/
  logs/run_log.txt
  plots/metrics.csv, *.png
src/
  algorithms.py       # BFS, UCS, A*, hill_climbing, simulated_annealing
  grid.py             # Grid loader, costs, dynamic obstacles, validity checks
  dynamic.py          # DynamicAgent with replanning
  cli.py              # Command-line interface
  plot_results.py     # Reads metrics.csv and generates plots
requirements.txt
README.md
```

## Installation

- Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows PowerShell:
. .venv/Scripts/Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

Run algorithms via the CLI. Maps are in `maps/`. Each run prints the start/goal, path summary, and writes logs/metrics to `results/`.

### Static algorithms (BFS, UCS, A\*)

- BFS

```bash
python -m src.cli --map maps/small.txt --algorithm bfs
```

- UCS

```bash
python -m src.cli --map maps/medium.txt --algorithm ucs
```

- A\*

```bash
python -m src.cli --map maps/large.txt --algorithm a_star
```

Metrics are logged to:

- Log file: `results/logs/run_log.txt`
- CSV: `results/plots/metrics.csv` with columns: Algorithm, Map, PathLength, Cost, NodesExpanded, Runtime

### Dynamic agent (replanning)

Dynamic runs use the agent, log detailed steps to `run_log.txt`, and also append limited metrics to `metrics.csv`:

- PathLength: measured by time steps taken
- Runtime: total time of the dynamic run
- Cost and NodesExpanded are not applicable and recorded as NA

- Hill Climbing

```bash
python -m src.cli --map maps/dynamic.txt --algorithm hill_climbing
```

- Simulated Annealing

```bash
python -m src.cli --map maps/dynamic.txt --algorithm simulated_annealing
```

Note: The CLI runs A\* in static mode when `--algorithm a_star` is provided. The dynamic agent strategies via CLI are `hill_climbing` and `simulated_annealing`.

## Maps

Map files are plain text. The first line contains six integers:

- rows cols start_x start_y goal_x goal_y

Subsequent lines are a grid where:

- `S` and `G` mark start/goal cells and count as cost 1
- `#` marks blocked cells (impassable)
- Digits `0-9` indicate traversal cost (e.g., `1` for normal)
- Any other character defaults to cost 1

Example:

```
5 5 0 0 4 4
S1111
11111
11#11
11111
1111G
```

A demo dynamic obstacle schedule is encoded in `src/grid.py`:

```python
self.dynamic_obstacles = {
    2: {(2, 2)},
    4: {(3, 3)},
}
```

## Results and plotting

After running static algorithms, generate plots from `results/plots/metrics.csv`:

```bash
python -m src.plot_results
```

This creates:

- `results/plots/runtime_comparison.png`
- `results/plots/nodes_expanded.png`
- `results/plots/path_cost.png`

The plotting script auto-detects whether the CSV has a header; no manual edits required.

## API overview

- `src.algorithms`
  - `bfs(grid, start, goal) -> (path, cost, nodes_expanded)`
  - `ucs(grid, start, goal) -> (path, cost, nodes_expanded)`
  - `a_star(grid, start, goal) -> (path, cost, nodes_expanded)`
  - `hill_climbing(grid, start, goal) -> path | None`
  - `simulated_annealing(grid, start, goal, ...) -> path | None`
- `src.grid`
  - `Grid(filename)` loads grid, costs, start/goal, and dynamic schedule
  - `is_valid(x, y, time=0)`, `get_cost(x, y)`
- `src.dynamic`
  - `DynamicAgent(grid, start, goal, strategy)` with `move()` and internal replanning

## Reproducibility and logs

- All runs append to `results/logs/run_log.txt` with timestamps and summary.
- Static metrics accumulate in `results/plots/metrics.csv`. You can append multiple runs for comparative plots.

## Troubleshooting

- No output plots: Ensure you’ve executed at least one static CLI run (BFS/UCS/A\*) to create `metrics.csv` before running `plot_results`.
- File not found: Use absolute or correct relative path for `--map`.
- Permission errors on Windows: Ensure the `results/` directory is writable; subdirectories are created automatically.

## License

See `LICENSE` for details.
