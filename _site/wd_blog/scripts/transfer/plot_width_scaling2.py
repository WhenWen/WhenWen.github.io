import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = "/afs/cs.stanford.edu/u/kaiyue/TPU/Marins/square_learning_rate/wandb_hyperball_runs2.csv"
OUTPUT_DIR = "/afs/cs.stanford.edu/u/kaiyue/TPU/Marins/square_learning_rate"


def make_plots(csv_path: str = CSV_PATH, output_dir: str = OUTPUT_DIR) -> None:
    df = pd.read_csv(csv_path)

    # Filter to exactly 4 layers
    df = df[(df["num_layers"] == 4.0) & (df["min_lr_ratio"] == 0.0) & (df["loss"] < 5.0) & (df["input_norm"] == False)].copy()

    # Ensure proper dtypes
    df["learning_rate"] = pd.to_numeric(df["learning_rate"], errors="coerce")
    df["loss"] = pd.to_numeric(df["loss"], errors="coerce")
    df["hidden_dim"] = pd.to_numeric(df["hidden_dim"], errors="coerce")

    df = df.dropna(subset=["learning_rate", "loss", "hidden_dim", "optimizer_name"])  # type: ignore[arg-type]

    os.makedirs(output_dir, exist_ok=True)

    for optimizer_name in ["muonh"]:
        sub = df[df["optimizer_name"] == optimizer_name].copy()
        if sub.empty:
            continue

        # One line per hidden_dim
        hidden_dims = sorted(sub["hidden_dim"].unique())
        # one line per warmup
        warmups = sorted(sub["warmup"].unique())

        plt.figure(figsize=(7, 4.5))
        for hd in hidden_dims:
            for warmup in warmups:
                d = sub[(sub["hidden_dim"] == hd) & (sub["warmup"] == warmup)].copy()
                if d.empty:
                    continue
                d = d.sort_values("learning_rate")
                plt.plot(
                    d["learning_rate"],
                    d["loss"],
                    marker="o",
                    linewidth=2,
                    markersize=5,
                    label=f"hidden_dim={int(hd) if float(hd).is_integer() else hd} warmup={warmup}",
                )

        # Add a star at the lowest loss point for each hidden_dim and connect with a dashed curve
        star_points = []
        for hd in hidden_dims:
            d = sub[sub["hidden_dim"] == hd]
            if d.empty:
                continue
            min_idx = d["loss"].idxmin()
            min_row = d.loc[min_idx]
            star_points.append((float(min_row["learning_rate"]), float(min_row["loss"]), min_row["hidden_dim"]) )
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
        plt.title(f"Loss vs LR (num_layers=4) — {optimizer_name}")
        plt.legend(title="hidden_dim", loc="best")
        plt.grid(True, which="both", linestyle=":", linewidth=0.6)
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"loss_vs_lr_layers4_{optimizer_name}.png")
        plt.savefig(out_path, dpi=200)
        # Also show for quick inspection
        plt.show()


if __name__ == "__main__":
    make_plots()
