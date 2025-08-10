# tools/wandb_topk.py

import argparse
import os
import wandb  # noqa
import pandas as pd  # noqa
from rich.console import Console  # noqa
from rich.table import Table  # noqa

# --- Globals / knobs ---
SIG_DIGITS = 8  # significant digits for all floats (scientific notation)
SCI_FMT_ALWAYS = True  # always use scientific notation for floats

# Display-only header renames (keeps internal keys the same)
DISPLAY_RENAMES = {
    "num_frames_per_video": "framespervid",
    "train_batchSize": "batchSize",
    "weight_decay": "weight_dec",
}

# Common prefixes to strip for cleaner display
CKPT_PREFIX_TO_STRIP = "gs://training-job-outputs/best_checkpoints/"
RUNNAME_PREFIX_TO_STRIP = "effort_"  # strip from displayed run_name only

# Prefer these keys; each value is a list of aliases we’ll try in order
KEY_ALIASES = {
    "lr": ["lr", "learning_rate", "optimizer.lr", "opt.lr", "train.lr"],
    "weight_decay": ["weight_decay", "wd", "optimizer.weight_decay", "opt.weight_decay"],
    "eps": ["eps", "optimizer_eps", "optimizer.eps", "opt.eps"],
    "train_batchSize": ["train_batchSize", "train.batch_size", "batch_size", "train.batchSize", "train.batchsize"],
    "num_frames_per_video": ["num_frames_per_video", "frames_per_video", "train.frames_per_video"],
}


def _flatten(d, parent_key: str = "", sep: str = "."):
    """Flatten a nested dict-like (including wandb.Config) into dot.keys."""
    items = {}
    try:
        iterable = d.items()
    except AttributeError:
        return items
    for k, v in iterable:
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if hasattr(v, "items"):
            items.update(_flatten(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def _get_param(flat_cfg, aliases, default="N/A"):
    """Return first value found for any alias in a flattened config."""
    for key in aliases:
        if key in flat_cfg:
            return flat_cfg[key]
        leaf = key.split(".")[-1]
        if leaf in flat_cfg:
            return flat_cfg[leaf]
    return default


def _fmt_num(x):
    """Format numbers: always scientific with SIG_DIGITS significant digits."""
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if SCI_FMT_ALWAYS:
            return f"{x:.{SIG_DIGITS}g}"
    return str(x)


def _strip_prefix(s: str, prefix: str):
    if isinstance(s, str) and s.startswith(prefix):
        return s[len(prefix):]
    return s


def _display_run_name(name: str):
    """Strip leading RUNNAME_PREFIX_TO_STRIP from display, leave rest intact."""
    return _strip_prefix(name, RUNNAME_PREFIX_TO_STRIP)


def _display_ckpt(path: str):
    """
    Return a single-line, prefix-stripped checkpoint path:
    e.g., 'bqgzrhve/ckpt_effort_2025...pth'
    """
    if not isinstance(path, str) or not path:
        return str(path)
    full = path.strip()
    return _strip_prefix(full, CKPT_PREFIX_TO_STRIP)


def analyze_sweep(project_name: str, sweep_id: str, k: int, metric: str, csv_path: str = None):
    """
    Analyzes a W&B sweep to find the top and bottom K runs based on a specified metric.

    Args:
        project_name (str): The W&B project name in the format 'entity/project' (or just 'project'
                            if your default entity is already set).
        sweep_id (str): The ID of the sweep to analyze (e.g., 'abc1234').
        k (int): The number of top and bottom runs to display.
        metric (str): The summary metric to use for sorting (e.g., 'val/best_metric').
        csv_path (str, optional): If provided, saves the full results to a CSV file.
    """
    console = Console()
    console.print(f"🔍 Analyzing sweep [bold cyan]{sweep_id}[/] in project [bold cyan]{project_name}[/]...")
    console.print(f"📈 Sorting by metric: [bold magenta]{metric}[/], K={k}")

    try:
        api = wandb.Api()
        sweep = api.sweep(f"{project_name}/{sweep_id}")
        runs = sorted(
            sweep.runs,
            key=lambda r: (r.state == "finished", r.summary.get(metric, float("-inf"))),
            reverse=True
        )
    except Exception as e:
        console.print(f"[bold red]Error accessing W&B API:[/bold red] {e}")
        return

    run_rows = []
    for run in runs:
        if run.state != "finished":
            continue

        if metric not in run.summary:
            console.print(
                f"⚠️ Run [yellow]{run.name}[/] (ID: {run.id}) missing metric '{metric}'.",
                style="yellow"
            )

        flat_cfg = _flatten(run.config)

        row = {
            "run_name": run.name,
            "metric": run.summary.get(metric, float("nan")),
            "best_ckpt_gcs": run.summary.get("best_ckpt_gcs", "N/A"),
        }

        for kparam, aliases in KEY_ALIASES.items():
            row[kparam] = _get_param(flat_cfg, aliases, default="N/A")

        run_rows.append(row)

    if not run_rows:
        console.print("[bold red]No finished runs found in the sweep.[/bold red]")
        return

    df = pd.DataFrame(run_rows)
    df = df.sort_values(by="metric", ascending=False, na_position="last").reset_index(drop=True)

    k = max(1, min(k, len(df)))
    display_columns = ["run_name", "metric"] + list(KEY_ALIASES.keys()) + ["best_ckpt_gcs"]

    # --- Print Top K Runs ---
    top_k_df = df.head(k)
    title_top = f"🏆 Top {k} Runs by '{metric}' — run names shown without leading '{RUNNAME_PREFIX_TO_STRIP}'"
    table = Table(title=title_top, style="green", title_style="bold green")
    for col in display_columns:
        header = DISPLAY_RENAMES.get(col, col)
        table.add_column(header, justify="left", overflow="fold", no_wrap=False)
    for _, row in top_k_df.iterrows():
        rn_disp = _display_run_name(str(row["run_name"]))
        vals = []
        for col in display_columns:
            if col == "run_name":
                vals.append(rn_disp)
            elif col == "best_ckpt_gcs":
                vals.append(_display_ckpt(row[col]))
            else:
                vals.append(_fmt_num(row[col]))
        table.add_row(*vals)
    Console().print(table)

    # --- Print Bottom K Runs ---
    bottom_k_df = df.tail(k)
    title_bot = f"📉 Bottom {k} Runs by '{metric}' — run names shown without leading '{RUNNAME_PREFIX_TO_STRIP}'"
    table = Table(title=title_bot, style="red", title_style="bold red")
    for col in display_columns:
        header = DISPLAY_RENAMES.get(col, col)
        table.add_column(header, justify="left", overflow="fold", no_wrap=False)
    for _, row in bottom_k_df.iterrows():
        rn_disp = _display_run_name(str(row["run_name"]))
        vals = []
        for col in display_columns:
            if col == "run_name":
                vals.append(rn_disp)
            elif col == "best_ckpt_gcs":
                vals.append(_display_ckpt(row[col]))
            else:
                vals.append(_fmt_num(row[col]))
        table.add_row(*vals)
    Console().print(table)

    # --- Optional: Save to CSV ---
    if csv_path:
        try:
            csv_df = df.rename(columns=DISPLAY_RENAMES).copy()
            csv_df.insert(0, "run_name_display", csv_df["run_name"].map(lambda s: _display_run_name(str(s))))
            # store a single-line, prefix-stripped checkpoint path in CSV too
            csv_df["best_ckpt_relative"] = csv_df["best_ckpt_gcs"].map(
                lambda s: _strip_prefix(str(s).strip(), CKPT_PREFIX_TO_STRIP)
            )
            csv_df.to_csv(csv_path, index=False)
            console.print(f"\n✅ Full results for {len(df)} runs saved to [bold cyan]{csv_path}[/]")
        except Exception as e:
            console.print(f"\n[bold red]Error saving CSV:[/bold red] {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze W&B sweep results.")
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="W&B project path in 'entity/project' or just 'project' (uses default entity)."
    )
    parser.add_argument(
        "--sweep_id",
        type=str,
        required=True,
        help="The ID of the sweep to analyze."
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of top and bottom runs to display."
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="val/best_metric",
        help="The summary metric to sort by."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="Optional path to save the full results as a CSV file."
    )

    args = parser.parse_args()
    analyze_sweep(args.project, args.sweep_id, args.k, args.metric, args.csv_path)
