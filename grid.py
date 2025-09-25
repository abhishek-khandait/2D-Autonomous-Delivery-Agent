class Grid:
    def __init__(self, filename):
        self.dynamic_obstacles = {}  # {time: set((x,y))}
        self.grid = self.load_grid(filename)
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])
        self.start_x, self.start_y, self.goal_x, self.goal_y = self.find_start_goal()
        print("Loaded Grid:")
        for row in self.grid:
            print(" ".join(map(str, row)))

    def load_grid(self, filename):
        grid = []
        with open(filename, "r") as f:
            dims = list(map(int, f.readline().strip().split()))
            (
                self.rows,
                self.cols,
                self.start_x,
                self.start_y,
                self.goal_x,
                self.goal_y,
            ) = dims[:6]
            for line in f:
                if line.strip():
                    row = []
                    for char in line.strip():
                        if char == "S":
                            row.append(1)
                        elif char == "G":
                            row.append(1)
                        elif char == "#":
                            row.append(float("inf"))
                        elif char.isdigit():
                            row.append(int(char))
                        else:
                            row.append(1)
                    grid.append(row)

        # Example: dynamic obstacle schedule (for demo only)
        # At t=2, block (2,2). At t=4, block (3,3).
        self.dynamic_obstacles = {
            2: {(2, 2)},
            4: {(3, 3)},
        }

        return grid

    def find_start_goal(self):
        return self.start_x, self.start_y, self.goal_x, self.goal_y

    def is_valid(self, x, y, time=0):
        if not (0 <= x < self.rows and 0 <= y < self.cols):
            return False
        if self.grid[x][y] == float("inf"):
            return False
        if (x, y) in self.dynamic_obstacles.get(time, set()):
            return False
        return True

    def get_cost(self, x, y):
        return (
            self.grid[x][y]
            if 0 <= x < self.rows and 0 <= y < self.cols
            else float("inf")
        )

    def print_grid(self):
        for row in self.grid:
            print(" ".join(map(str, row)))
