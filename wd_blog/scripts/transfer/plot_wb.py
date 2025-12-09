import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional
import pandas as pd
import wandb
import tqdm

def get_nested(config: Dict[str, Any], path: Iterable[str], default: Any = None) -> Any:
    current: Any = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _traverse_nested(d: Dict[str, Any], path: Iterable[str]) -> Any:
    current: Any = d
    for part in path:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _coerce_summary_to_dict(summary_obj: Any) -> Dict[str, Any]:
    # Try common representations of the W&B Summary object into a plain dict
    if isinstance(summary_obj, dict):
        return summary_obj
    try:
        return dict(summary_obj)  # may work if it's Mapping-like
    except Exception:
        pass
    try:
        raw = getattr(summary_obj, "_json_dict", None)
        if isinstance(raw, dict):
            return raw
    except Exception:
        pass
    return {}


def get_final_metric_from_summary(run: "wandb.apis.public.Run", key: str) -> Optional[float]:
    # First, attempt direct access which handles flattened keys in many projects
    try:
        value = run.summary.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    except Exception:
        # Fall through to dict-based attempts
        pass

    # Coerce to dict and try both flat and nested forms
    summary_dict = _coerce_summary_to_dict(getattr(run, "summary", {}))

    # Flat key
    value = summary_dict.get(key)
    if isinstance(value, (int, float)):
        return float(value)

    # Nested path using '/' separators
    if "/" in key:
        nested = _traverse_nested(summary_dict, key.split("/"))
        if isinstance(nested, (int, float)):
            return float(nested)

    return None


def get_final_metric_from_history(run: "wandb.apis.public.Run", key: str) -> Optional[float]:
    try:
        # Pull only the requested key to reduce transfer; take the last non-null
        history_df = run.history(keys=[key], pandas=True)
        if key in history_df.columns:
            series = history_df[key].dropna()
            if not series.empty:
                return float(series.iloc[-1])
    except Exception:
        # Let callers try alternative keys if needed
        return None
    return None


def _run_to_row(run: "wandb.apis.public.Run") -> Optional[Dict[str, Any]]:
    try:
        # W&B public API's run.config may not be a plain dict; guard conversion
        raw_cfg = run.config or {}
        cfg = raw_cfg

        # Extract requested config fields (robust to missing keys)
        model_hidden_dim = get_nested(cfg, ["model", "hidden_dim"])  # 1
        optimizer_lr = get_nested(cfg, ["optimizer", "learning_rate"])  # 2
        optimizer_min_lr_ratio = get_nested(cfg, ["optimizer", "min_lr_ratio"])  # 3
        optimizer_adam_lr = get_nested(cfg, ["optimizer", "adam_lr"])  # 4
        model_num_layers = get_nested(cfg, ["model", "num_layers"])  # 6
        warmup = get_nested(cfg, ["optimizer", "warmup"])  # 7

        run_name = run.name
        if "muonh" in run_name:
            optimizer_name = "muonh"
        elif "adamh" in run_name:
            optimizer_name = "adamh"
        elif "muon-" in run_name:
            optimizer_name = "muon"
        elif ("adamw-" in run_name) or ("adam-" in run_name):
            optimizer_name = "adam"
        else:
            print(f"Unknown optimizer name: {run_name}")
            optimizer_name = None

        metric_keys_to_try = [
            "eval/paloma/c4_en/loss",
        ]

        final_c4_en_loss: Optional[float] = None
        for metric_key in metric_keys_to_try:
            final_c4_en_loss = get_final_metric_from_history(run, metric_key)
            if final_c4_en_loss is not None:
                break

        row: Dict[str, Any] = {
            "run_id": run.id,
            "run_name": run_name,  # 5
            "hidden_dim": model_hidden_dim,  # 1
            "learning_rate": optimizer_lr,  # 2
            "min_lr_ratio": optimizer_min_lr_ratio,  # 3
            "adam_lr": optimizer_adam_lr,  # 4
            "num_layers": model_num_layers,  # 6
            "loss": final_c4_en_loss,
            "optimizer_name": optimizer_name,
            "warmup": warmup,
        }
        return row
    except Exception as exc:
        print(f"Failed to process run {getattr(run, 'id', 'unknown')}: {exc}")
        return None


def fetch_runs_dataframe(project_path: str, max_workers: Optional[int] = None) -> pd.DataFrame:
    api = wandb.Api(timeout=60)
    runs = list(api.runs(project_path))

    if not runs:
        return pd.DataFrame()

    if max_workers is None:
        cpu_count = os.cpu_count() or 4
        max_workers = min(32, cpu_count + 4)

    rows: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_run = {executor.submit(_run_to_row, run): run for run in runs}
        progress = tqdm.tqdm(
            as_completed(future_to_run),
            total=len(future_to_run),
            desc="Processing W&B runs",
        )
        for future in progress:
            row = future.result()
            if row is not None:
                rows.append(row)

    df = pd.DataFrame(rows)
    return df


def main() -> None:
    # Project from: https://wandb.ai/marin-community/Hyperball
    project_path = "marin-community/Hyperball"
    df = fetch_runs_dataframe(project_path)

    # Save results next to this script
    out_csv = os.path.join(os.path.dirname(__file__), "wandb_hyperball_runs.csv")
    df.to_csv(out_csv, index=False)

    # Print a small preview
    with pd.option_context("display.max_columns", None):
        print(df.head(20))
    print(f"Saved {len(df)} rows to {out_csv}")


if __name__ == "__main__":
    main()


