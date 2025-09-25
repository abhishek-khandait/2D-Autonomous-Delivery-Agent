import argparse
import os
import time
from datetime import datetime
from src.grid import Grid
from src.algorithms import bfs, ucs, a_star
from src.dynamic import DynamicAgent
import csv

LOG_FILE = os.path.join("results", "logs", "run_log.txt")
CSV_FILE = os.path.join("results", "plots", "metrics.csv")


def ensure_log_dir():
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)


def log_run(algorithm, mapfile, path=None, cost=None, runtime=None, dynamic=False):
    """Append results to run_log.txt"""
    ensure_log_dir()
    with open(LOG_FILE, "a") as f:
        f.write(
            f"\n--- Run [{datetime.now()}] | Algo: {algorithm} | Map: {mapfile} ---\n"
        )
        if path:
            f.write(f"Path length: {len(path)}\n")
            f.write(f"Path: {path}\n")
        else:
            f.write("No path found\n")
        if cost is not None:
            f.write(f"Total cost: {cost}\n")
        if runtime is not None:
            f.write(f"Runtime: {runtime:.6f} seconds\n")
        if dynamic:
            f.write("Dynamic replanning was enabled.\n")


def log_metrics(algorithm, mapfile, path, cost, nodes, runtime):
    os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                ["Algorithm", "Map", "PathLength", "Cost", "NodesExpanded", "Runtime"]
            )
        writer.writerow(
            [
                algorithm,
                os.path.basename(mapfile),
                len(path) if path else 0,
                cost if cost is not None else "NA",
                nodes if nodes is not None else "NA",
                f"{runtime:.6f}",
            ]
        )


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Delivery Agent: Run pathfinding algorithms on a grid."
    )
    parser.add_argument("--map", type=str, required=True, help="Path to map file")
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["bfs", "ucs", "a_star", "hill_climbing", "simulated_annealing"],
        required=True,
        help="Algorithm to use",
    )
    args = parser.parse_args()

    grid = Grid(args.map)
    start = (grid.start_x, grid.start_y)
    goal = (grid.goal_x, grid.goal_y)

    print("Start:", start)
    print("Goal:", goal)

    # --- Static algorithms ---
    if args.algorithm == "bfs":
        t0 = time.perf_counter()
        path, cost, nodes = bfs(grid, start, goal)
        runtime = time.perf_counter() - t0
        print("Path:", path if path else "No path")
        print("Cost:", cost, "Nodes expanded:", nodes, "Runtime:", runtime)
        log_run("bfs", args.map, path, cost, runtime)
        log_metrics("bfs", args.map, path, cost, nodes, runtime)

    elif args.algorithm == "ucs":
        t0 = time.perf_counter()
        path, cost, nodes = ucs(grid, start, goal)
        runtime = time.perf_counter() - t0
        print("Path:", path if path else "No path")
        print("Cost:", cost, "Nodes expanded:", nodes, "Runtime:", runtime)
        log_run("ucs", args.map, path, cost, runtime)
        log_metrics("ucs", args.map, path, cost, nodes, runtime)

    elif args.algorithm == "a_star":
        t0 = time.perf_counter()
        path, cost, nodes = a_star(grid, start, goal)
        runtime = time.perf_counter() - t0
        print("Path:", path if path else "No path")
        print("Cost:", cost, "Nodes expanded:", nodes, "Runtime:", runtime)
        log_run("a_star", args.map, path, cost, runtime)
        log_metrics("a_star", args.map, path, cost, nodes, runtime)

    # --- Dynamic algorithms ---
    else:
        agent = DynamicAgent(grid, start, goal, strategy=args.algorithm)
        t0 = time.perf_counter()
        success = agent.move()
        runtime = time.perf_counter() - t0
        # After move(), agent.path is consumed; use time_step as an effective path length
        dynamic_path_length = agent.time_step
        # Construct a placeholder path of the measured length for CSV logging compatibility
        placeholder_path = [None] * dynamic_path_length
        log_run(
            args.algorithm,
            args.map,
            path=placeholder_path if success else None,
            runtime=runtime,
            dynamic=True,
        )
        # Log limited metrics for dynamic strategies: PathLength and Runtime; Cost/Nodes are not applicable
        log_metrics(
            args.algorithm,
            args.map,
            placeholder_path if success else None,
            cost=None,
            nodes=None,
            runtime=runtime,
        )


if __name__ == "__main__":
    main()
