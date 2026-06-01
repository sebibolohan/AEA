import subprocess
import os

INSTANCES = {
    "myciel3": ("../aco/data/raw/myciel3.col", 4, 20),
    "myciel4": ("../aco/data/raw/myciel4.col", 5, 20),
    "queen5_5": ("../aco/data/raw/queen5_5.col", 5, 30),
    "queen7_7": ("../aco/data/raw/queen7_7.col", 7, 30),
    "miles1500": ("../aco/data/raw/miles1500.col", 73, 60),
    "fpsol2_i_1": ("../aco/data/raw/fpsol2.i.1.col", 65, 60),
    "inithx_i_1": ("../aco/data/raw/inithx.i.1.col", 54, 90),
    "flat300_28_0": ("../aco/data/raw/flat300_28_0.col", 28, 300),
    "le450_5a": ("../aco/data/raw/le450_5a.col", 5, 300),
    "DSJC125_9": ("../aco/data/raw/DSJC125.9.col", 44, 300),
    "DSJC500_1": ("../aco/data/raw/DSJC500.1.col", 12, 300),
    "DSJC500_5": ("../aco/data/raw/DSJC500.5.col", 48, 600),
    "DSJC500_9": ("../aco/data/raw/DSJC500.9.col", 126, 600),
    "DSJC1000_9": ("../aco/data/raw/DSJC1000.9.col", 223, 900),
}

os.makedirs("results", exist_ok=True)

csv_path = "results/partialcol_advanced_all.csv"

if os.path.exists(csv_path):
    os.remove(csv_path)

for name, (path, k, time_limit) in INSTANCES.items():
    print(f"\n=== {name} | k={k} | time={time_limit}s ===")

    cmd = [
        "python",
        "partialcol_advanced.py",
        path,
        "--k", str(k),
        "--time-limit", str(time_limit),
        "--runs", "5",
        "--csv", csv_path,
    ]

    subprocess.run(cmd, check=False)