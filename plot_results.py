import pandas as pd
import matplotlib.pyplot as plt
import os

# Path to metrics file
csv_file = os.path.join("results", "plots", "metrics.csv")

expected_cols = ["Algorithm", "Map", "PathLength", "Cost", "NodesExpanded", "Runtime"]

# Ensure plots directory exists
os.makedirs("results/plots", exist_ok=True)

# Read CSV whether it has a header or not
if not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0:
    raise FileNotFoundError(
        "metrics.csv not found or empty. Run the CLI to generate metrics before plotting."
    )

with open(csv_file, "r", encoding="utf-8") as f:
    first_line = f.readline().strip()

has_header = all(h in first_line.split(",") for h in ["Algorithm", "Map"]) and (
    "," in first_line or first_line == ";".join(expected_cols)
)

if has_header:
    df = pd.read_csv(csv_file)
else:
    df = pd.read_csv(csv_file, header=None, names=expected_cols)

# Coerce numeric columns
for col in ["PathLength", "Cost", "NodesExpanded", "Runtime"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# --- Runtime Comparison ---
plt.figure()
for algo in df["Algorithm"].dropna().unique():
    subset = df[df["Algorithm"] == algo]
    plt.plot(subset["Map"], subset["Runtime"], marker="o", label=algo)
plt.xlabel("Map")
plt.ylabel("Runtime (s)")
plt.title("Runtime Comparison")
plt.legend()
plt.savefig("results/plots/runtime_comparison.png")

# --- Nodes Expanded ---
plt.figure()
for algo in df["Algorithm"].dropna().unique():
    subset = df[df["Algorithm"] == algo]
    plt.plot(subset["Map"], subset["NodesExpanded"], marker="o", label=algo)
plt.xlabel("Map")
plt.ylabel("Nodes Expanded")
plt.title("Nodes Expanded Comparison")
plt.legend()
plt.savefig("results/plots/nodes_expanded.png")

# --- Path Cost ---
plt.figure()
for algo in df["Algorithm"].dropna().unique():
    subset = df[df["Algorithm"] == algo]
    plt.plot(subset["Map"], subset["Cost"], marker="o", label=algo)
plt.xlabel("Map")
plt.ylabel("Path Cost")
plt.title("Path Cost Comparison")
plt.legend()
plt.savefig("results/plots/path_cost.png")

print("✅ Plots generated in results/plots/")
