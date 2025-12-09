import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = "/afs/cs.stanford.edu/u/kaiyue/TPU/Marins/square_learning_rate/wandb_hyperball_runs2.csv"
OUTPUT_DIR = "/afs/cs.stanford.edu/u/kaiyue/TPU/Marins/square_learning_rate"


def make_plots(csv_path: str = CSV_PATH, output_dir: str = OUTPUT_DIR) -> None:
    df = pd.read_csv(csv_path)

    # Filter to exactly hidden_dim = 128 and min_lr_ratio = 0.0
    df = df[(df["hidden_dim"] == 128.0) & (df["min_lr_ratio"] == 0.0) & (df["loss"] < 4.6) & (df["input_norm"] == False)].copy()

    # Ensure proper dtypes
    df["learning_rate"] = pd.to_numeric(df["learning_rate"], errors="coerce")
    df["loss"] = pd.to_numeric(df["loss"], errors="coerce")
    df["num_layers"] = pd.to_numeric(df["num_layers"], errors="coerce")

    df = df.dropna(subset=["learning_rate", "loss", "num_layers", "optimizer_name"])  # type: ignore[arg-type]

    os.makedirs(output_dir, exist_ok=True)

    for optimizer_name in ["muonh"]:
        sub = df[df["optimizer_name"] == optimizer_name].copy()
        print(sub)
        if sub.empty:
            continue

        # One line per num_layers
        depths = sorted(sub["num_layers"].unique())

        plt.figure(figsize=(7, 4.5))
        for depth in depths:
            d = sub[sub["num_layers"] == depth].copy()
            d = d.sort_values("learning_rate")
            d = d[(d["learning_rate"] > 5e-3) & (d["learning_rate"] < 2e-2)]
            plt.plot(
                d["learning_rate"],
                d["loss"],
                marker="o",
                linewidth=2,
                markersize=5,
                label=f"num_layers={int(depth) if float(depth).is_integer() else depth}",
            )

        # Add a star at the lowest loss point for each depth and connect with a dashed curve
        star_points = []
        for depth in depths:
            d = sub[sub["num_layers"] == depth]
            if d.empty:
                continue
            min_idx = d["loss"].idxmin()
            min_row = d.loc[min_idx]
            star_points.append((float(min_row["learning_rate"]), float(min_row["loss"]), min_row["num_layers"]))
            plt.plot(
                min_row["learning_rate"],
                min_row["loss"],
                marker="*",
                markersize=12,
                markerfacecolor="none",
                markeredgecolor="k",
                linestyle="none",
                label="_nolegend_",
                zorder=5,
            )

        if star_points:
            star_points_sorted = sorted(star_points, key=lambda t: t[1])
            xs = np.array([p[0] for p in star_points_sorted])
            ys = np.array([p[1] for p in star_points_sorted])
            plt.plot(
                xs,
                ys,
                linestyle="--",
                color="k",
                linewidth=1.5,
                label="min-loss envelope",
            )

        plt.xscale("log")
        plt.xlabel("learning rate")
        plt.ylabel("loss")
        plt.title(f"Loss vs LR (hidden_dim=128) — {optimizer_name}")
        plt.legend(title="num_layers", loc="best")
        plt.grid(True, which="both", linestyle=":", linewidth=0.6)
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"loss_vs_lr_hidden128_{optimizer_name}2.png")
        plt.savefig(out_path, dpi=200)
        # Also show for quick inspection
        plt.show()


if __name__ == "__main__":
    make_plots()


