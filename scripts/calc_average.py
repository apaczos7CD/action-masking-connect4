from pathlib import Path
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate all Connect4 checkpoints vs uniform-random-legal opponent."
    )
    parser.add_argument(
        "--algo",
        type=str,
        choices=["ppo", "maskable_ppo"],
        default="ppo",
        help="Algorithm used in checkpoint filenames.",
    )
    return parser.parse_args()

args = parse_args()

# Folder z plikami wejściowymi
results_dir = Path(r"C:\Users\apacz\PycharmProjects\Inzynierka\results")

# Lista plików wejściowych
files = [
    results_dir / f"eval_{args.algo}_seed0.csv",
    results_dir / f"eval_{args.algo}_seed1.csv",
    results_dir / f"eval_{args.algo}_seed2.csv",
]

# Wczytanie i połączenie wszystkich plików
dfs = [pd.read_csv(file) for file in files]
df = pd.concat(dfs, ignore_index=True)

# Obliczenie średnich dla każdego kroku
avg_df = (
    df.groupby("step", as_index=False)[["win_rate", "ci_low", "ci_high"]]
    .mean()
    .sort_values("step")
)

# Dodanie informacji o algorytmie
avg_df["algo"] = args.algo

# Zapis do pliku CSV w folderze results
output_file = results_dir / f"eval_{args.algo}_average.csv"
avg_df.to_csv(output_file, index=False)

print(f"Zapisano plik: {output_file}")