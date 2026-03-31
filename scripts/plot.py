import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Print plots for train process for both algorithms."
    )
    parser.add_argument(
        "--file1",
        type=str,
        default="eval_maskable_ppo_seed0_evalseed0.csv",
        help="File 1 path.",
    )
    parser.add_argument(
        "--file2",
        type=str,
        default="eval_ppo_seed0_evalseed0.csv",
        help="File 2 path.",
    )
    return parser.parse_args()

args = parse_args()

# Katalog główny projektu
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

# Nazwy plików CSV
file_1 = RESULTS_DIR / args.file1
file_2 = RESULTS_DIR / args.file2

# Wczytanie danych
df1 = pd.read_csv(file_1)
df2 = pd.read_csv(file_2)

# Połączenie w jedną ramkę
df = pd.concat([df1, df2], ignore_index=True)

# Rysowanie wykresu
plt.figure(figsize=(10, 6))

for algo_name, group in df.groupby("algo"):
    group = group.sort_values(by="step")
    plt.plot(group["step"], group["wins"], marker="o", label=algo_name)

plt.xlabel("step")
plt.ylabel("wins")
plt.title("Wins vs step")
plt.legend(title="algo")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()