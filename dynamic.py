import os
from datetime import datetime
from .algorithms import hill_climbing, simulated_annealing, a_star

LOG_FILE = os.path.join("results", "logs", "run_log.txt")


class DynamicAgent:
    def __init__(self, grid, start, goal, strategy="hill_climbing"):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.strategy = strategy
        self.time_step = 0
        self.path = self.plan_path(start)

        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(
                f"\n--- New Run [{datetime.now()}] | Strategy: {self.strategy} ---\n"
            )

    def plan_path(self, start):
        """Compute a path from `start` to goal using the chosen strategy"""
        if self.strategy == "a_star":
            path, _, _ = a_star(self.grid, start, self.goal)
        elif self.strategy == "hill_climbing":
            path = hill_climbing(self.grid, start, self.goal)
        elif self.strategy == "simulated_annealing":
            path = simulated_annealing(self.grid, start, self.goal)
        else:
            raise ValueError("Unknown strategy")
        return path

    def move(self):
        """Execute moves step by step, replanning if obstacles appear"""
        while self.path:
            current = self.path.pop(0)

            # Check if current cell is still valid (dynamic obstacles)
            if not self.grid.is_valid(*current, time=self.time_step):
                msg = f"[t={self.time_step}] Obstacle detected at {current}. Replanning..."
                print(msg)
                with open(LOG_FILE, "a") as f:
                    f.write(msg + "\n")

                self.path = self.plan_path(current)
                if not self.path:
                    msg = f"[t={self.time_step}] No path found. Delivery failed!"
                    print(msg)
                    with open(LOG_FILE, "a") as f:
                        f.write(msg + "\n")
                    return False
            else:
                msg = f"[t={self.time_step}] Moving to {current}"
                print(msg)
                with open(LOG_FILE, "a") as f:
                    f.write(msg + "\n")

            self.time_step += 1  # advance time

        msg = f"[t={self.time_step}] Package delivered successfully!"
        print(msg)
        with open(LOG_FILE, "a") as f:
            f.write(msg + "\n")
        return True
