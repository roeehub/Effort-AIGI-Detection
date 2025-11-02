import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table

# --- Configuration: This section defines the entire strategy ---

# The OOF scores file from the fusion run
FILE_PATH = "out_full/fusion_oof_scores.csv"

# The two thresholds determined for the "Supported Methods" strategy
T_LOW = 0.964434
T_HIGH = 0.983175

# Methods to be excluded from the primary dataset entirely
DATASET_EXCLUSIONS = ['faceforensics++']

# Fake methods to be treated as "Unsupported"
UNSUPPORTED_METHODS = ['veo3_creations', 'sadtalker', 'deepfakedetection']


def classify_video(score: float) -> str:
    """Applies the three-way decision logic."""
    if score < T_LOW:
        return "REAL (Miss)"
    elif score > T_HIGH:
        return "FAKE (Caught)"
    else:
        return "UNCERTAIN"


def generate_real_method_summary(df_base, console):
    """Generates and prints a per-method breakdown for REAL videos."""

    real_method_table = Table(title="[bold]Per-Method Breakdown (Real Videos)[/bold]", style="green", title_justify="left")
    real_method_table.add_column("Method", style="bold magenta")
    real_method_table.add_column("Video Count", style="cyan", justify="right")
    real_method_table.add_column("% Correct (as REAL)", justify="right")
    real_method_table.add_column("% UNCERTAIN", justify="right")
    real_method_table.add_column("FPR (Caught as FAKE)", justify="right")

    df_reals = df_base[df_base['label'] == 0].copy()

    if df_reals.empty:
        console.print("[yellow]No real videos found to generate a per-method breakdown.[/yellow]")
        return

    # Calculate percentages using normalize=True
    method_percs = df_reals.groupby('method')['classification'].value_counts(normalize=True).unstack(fill_value=0)

    # Calculate total counts for each method
    total_counts = df_reals['method'].value_counts().rename('count')

    # Safely join the percentages and counts into a single DataFrame
    summary_df = method_percs.join(total_counts)

    # In case a method had videos but all were filtered somehow, fill NaN in count with 0
    summary_df['count'].fillna(0, inplace=True)
    summary_df['count'] = summary_df['count'].astype(int)


    # Sort by method name
    summary_df = summary_df.sort_index()

    for method, data in summary_df.iterrows():
        # Because we used normalize=True, the values are fractions (e.g., 0.94). Multiply by 100 for display.
        correct_perc = data.get('REAL (Miss)', 0) * 100
        uncertain_perc = data.get('UNCERTAIN', 0) * 100
        fpr_perc = data.get('FAKE (Caught)', 0) * 100

        real_method_table.add_row(
            method,
            f"{data['count']}", # Now this is a guaranteed integer
            f"{correct_perc:.1f}%",
            f"{uncertain_perc:.1f}%",
            f"{fpr_perc:.1f}%"
        )

    console.print(real_method_table)


def generate_summary():
    """Loads data, performs calculations, and prints summary tables."""

    console = Console()
    console.print("[bold cyan]Running Final Performance Verification...[/bold cyan]")

    # 1. Load and prepare the data
    try:
        df = pd.read_csv(FILE_PATH)
    except FileNotFoundError:
        console.print(
            f"[bold red]Error: Could not find the file '{FILE_PATH}'. Please make sure it's in the same directory.[/bold red]")
        return

    df_base = df[~df['method'].isin(DATASET_EXCLUSIONS)].copy()

    # 2. Add classification and category columns for easy grouping
    df_base['classification'] = df_base['noisy_or'].apply(classify_video)

    def get_category(row):
        if row['label'] == 0:
            return "Real Videos"
        elif row['method'] in UNSUPPORTED_METHODS:
            return "Unsupported Fakes"
        else:
            return "Supported Fakes"

    df_base['category'] = df_base.apply(get_category, axis=1)

    # 3. --- Calculations for the Executive Summary Table ---
    summary_data = {}

    # Group by our defined categories
    grouped = df_base.groupby('category')['classification'].value_counts(normalize=True).unstack(fill_value=0)

    # Calculate Overall performance for all fakes combined
    df_fakes_only = df_base[df_base['label'] == 1]
    overall_fake_perf = df_fakes_only['classification'].value_counts(normalize=True)

    # 4. --- Create and print the rich tables ---

    # Executive Summary Table
    exec_table = Table(title="[bold]Executive Performance Summary[/bold]", style="cyan", title_justify="left")
    exec_table.add_column("Category", style="bold magenta")
    exec_table.add_column("TPR (Caught as FAKE)", justify="right")
    exec_table.add_column("% UNCERTAIN", justify="right")
    exec_table.add_column("% Miss (Classified as REAL)", justify="right")

    def add_fake_row(table, name, data):
        table.add_row(
            name,
            f"{data.get('FAKE (Caught)', 0):.2%}",
            f"{data.get('UNCERTAIN', 0):.2%}",
            f"{data.get('REAL (Miss)', 0):.2%}"
        )

    add_fake_row(exec_table, "Performance on Supported Fakes", grouped.loc['Supported Fakes'])
    add_fake_row(exec_table, "Incidental Performance on Unsupported Fakes", grouped.loc['Unsupported Fakes'])
    exec_table.add_section()
    add_fake_row(exec_table, "Overall Performance (All Fakes Combined)", overall_fake_perf)

    # Calculate FPR on 'Certain' real videos
    reals_df = df_base[df_base['category'] == 'Real Videos']
    reals_certain = reals_df[reals_df['classification'] != 'UNCERTAIN']
    fpr = (reals_certain['classification'] == 'FAKE (Caught)').mean()

    # Add Real video performance row
    exec_table.add_section()
    exec_table.add_row(
        "Performance on Real Videos",
        f"[bold red]FPR: {fpr:.2%}[/bold red]",  # FPR
        f"{grouped.loc['Real Videos'].get('UNCERTAIN', 0):.2%}",
        f"{grouped.loc['Real Videos'].get('REAL (Miss)', 0):.2%} (Correct)"
    )

    console.print(exec_table)

    # Per-Method Breakdown for Real Videos
    generate_real_method_summary(df_base, console)

    # Per-Method Breakdown Table
    per_method_table = Table(title="[bold]Per-Method Breakdown[/bold]", style="cyan", title_justify="left")
    per_method_table.add_column("Method", style="bold magenta")
    per_method_table.add_column("Category", style="yellow")
    per_method_table.add_column("TPR (Caught)", justify="right")
    per_method_table.add_column("% Uncertain", justify="right")
    per_method_table.add_column("% Missed", justify="right")

    df_fakes = df_base[df_base['label'] == 1].copy()
    method_grouped = df_fakes.groupby('method')['classification'].value_counts(normalize=True).unstack(fill_value=0)

    # Add unsupported methods first
    for method in UNSUPPORTED_METHODS:
        if method in method_grouped.index:
            data = method_grouped.loc[method]
            per_method_table.add_row(
                method, "Unsupported", f"{data.get('FAKE (Caught)', 0):.1%}",
                f"{data.get('UNCERTAIN', 0):.1%}", f"{data.get('REAL (Miss)', 0):.1%}"
            )

    per_method_table.add_section()

    # Add supported methods, sorted alphabetically
    supported_methods_in_data = sorted([m for m in method_grouped.index if m not in UNSUPPORTED_METHODS])
    for method in supported_methods_in_data:
        data = method_grouped.loc[method]
        per_method_table.add_row(
            method, "Supported", f"{data.get('FAKE (Caught)', 0):.1%}",
            f"{data.get('UNCERTAIN', 0):.1%}", f"{data.get('REAL (Miss)', 0):.1%}"
        )

    console.print(per_method_table)


if __name__ == "__main__":
    generate_summary()
