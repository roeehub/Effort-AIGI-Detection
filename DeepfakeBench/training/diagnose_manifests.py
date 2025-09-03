#!/usr/bin/env python3
"""
diagnose_manifests.py

A comprehensive diagnostic tool that compares the source manifest (frame_manifest.json)
against the processed manifest (frame_properties.parquet).

It generates a detailed report breaking down:
- Frame counts before and after processing for each method.
- The percentage of frames dropped during processing.
- A comparison of raw video folder counts vs. unified video_id counts.

This is the primary tool for identifying data loss and diagnosing issues with
the ID unification logic in 'create_property_manifest.py'.

Usage:
    pip install rich pandas pyarrow
    python diagnose_manifests.py [--include-validation]
"""
import json
import argparse
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

# Methods are categorized based on the provided YAML structure.
# This makes the logic for calculating totals clear and maintainable.

# Methods considered to be 'real' sources
REAL_METHODS = [
    "Celeb-real",
    "YouTube-real",
    "external_youtube_avspeech",
    # NOTE: "FaceForensics++" is often treated as a real source for its fakes,
    # but isn't a standalone real method in the same way. We'll categorize it
    # based on its appearance in the manifests (e.g. 'ff-real'). Add if needed.
]

# Fake methods used for the training set
TRAINING_FAKE_METHODS = [
    # Face-swapping (FS)
    "simswap", "mobileswap", "faceswap", "inswap", "blendface", "fsgan", "uniface",
    # Face-reenactment (FR)
    "pirender", "facevid2vid", "lia", "fomm", "MRAA", "wav2lip", "mcnet", "danet",
    # Entire Face Synthesis (EFS)
    "VQGAN", "StyleGAN3", "StyleGAN2", "SiT", "RDDM", "ddim",
]

# Fake methods used for the validation (hold-out) set
VALIDATION_METHODS = [
    "facedancer",
    "sadtalker",
    "DiT",
    "StyleGANXL",
    "e4s",
    "one_shot_free",
]


def parse_source_manifest(manifest_path: Path) -> pd.DataFrame:
    """Parses the source JSON manifest and aggregates counts by method."""
    console = Console()
    console.print(f"Parsing source manifest: '{manifest_path}'...")
    with open(manifest_path, 'r') as f:
        all_paths = json.load(f)

    counts = {}
    for path_str in all_paths:
        try:
            # Assumes a path structure like: .../method/video_id/frame.jpg
            parts = Path(path_str).parts
            method, video_id = parts[-3], parts[-2]
        except IndexError:
            continue

        if method not in counts:
            counts[method] = {'frames': 0, 'videos': set()}

        counts[method]['frames'] += 1
        counts[method]['videos'].add(video_id)

    # Convert to a list of records for DataFrame creation
    records = []
    for method, data in counts.items():
        records.append({
            'Method': method,
            'Source_Frames': data['frames'],
            'Source_Videos': len(data['videos'])
        })

    if not records:
        return pd.DataFrame(columns=['Method', 'Source_Frames', 'Source_Videos'])

    return pd.DataFrame(records).sort_values('Method')


def parse_processed_manifest(manifest_path: Path) -> pd.DataFrame:
    """Parses the processed Parquet manifest and aggregates counts by method."""
    console = Console()
    console.print(f"Parsing processed manifest: '{manifest_path}'...")
    df = pd.read_parquet(manifest_path)

    # Use .size() which is robust and doesn't depend on a specific column name.
    analysis = df.groupby('method').agg(
        Processed_Frames=('video_id', 'size'),
        Processed_Unique_IDs=('video_id', 'nunique')
    ).reset_index()
    analysis.rename(columns={'method': 'Method'}, inplace=True)

    return analysis.sort_values('Method')


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Diagnose and compare source and processed data manifests."
    )
    parser.add_argument(
        '--include-validation',
        action='store_true',
        help="Include validation methods in the report."
    )
    args = parser.parse_args()

    console = Console()
    source_manifest_path = Path("frame_manifest.json")
    processed_manifest_path = Path("frame_properties.parquet")

    if not source_manifest_path.exists() or not processed_manifest_path.exists():
        console.print("[bold red]Error:[/] One or both manifest files not found.")
        console.print(f"  - Looked for '{source_manifest_path}'")
        console.print(f"  - Looked for '{processed_manifest_path}'")
        return

    source_df = parse_source_manifest(source_manifest_path)
    processed_df = parse_processed_manifest(processed_manifest_path)

    # Perform an outer merge to keep all methods from both files
    report_df = pd.merge(source_df, processed_df, on='Method', how='outer')
    report_df.fillna(0, inplace=True)  # Replace NaNs with 0 for methods missing in one file

    # Conditionally filter out validation methods
    if not args.include_validation:
        report_df = report_df[~report_df['Method'].isin(VALIDATION_METHODS)].copy()
        console.print(
            "[bold yellow]Note:[/] Validation methods are excluded. Use `--include-validation` to see them.\n"
        )

    # Convert counts to integers for clean display
    count_cols = [col for col in report_df.columns if col != 'Method']
    report_df[count_cols] = report_df[count_cols].astype(int)

    # Calculate drop percentage, handling division by zero
    report_df['Frames_Dropped_%'] = 100 * (
            (report_df['Source_Frames'] - report_df['Processed_Frames']) / report_df['Source_Frames'].replace(0, 1)
    )

    # --- Generate Rich Table ---
    table = Table(
        title="Comprehensive Manifest Diagnosis Report",
        show_header=True,
        header_style="bold magenta"
    )
    table.add_column("Method", style="cyan", no_wrap=True)
    table.add_column("Source Frames", justify="right", style="white")
    table.add_column("Processed Frames", justify="right", style="yellow")
    table.add_column("Frames Dropped %", justify="right", style="red")
    table.add_column("Source Videos (Raw Folders)", justify="right", style="white")
    table.add_column("Processed IDs (Unified)", justify="right", style="green")

    report_df = report_df.sort_values('Source_Frames', ascending=False)

    for _, row in report_df.iterrows():
        drop_percent_str = f"{row['Frames_Dropped_%']:.2f}%" if row['Source_Frames'] > 0 else "N/A"
        table.add_row(
            row['Method'],
            f"{row['Source_Frames']:,}",
            f"{row['Processed_Frames']:,}",
            drop_percent_str,
            f"{row['Source_Videos']:,}",
            f"{row['Processed_Unique_IDs']:,}"
        )

    console.print(table)

    # --- Calculate and Print Totals (LOGIC FIXED) ---
    # These calculations are performed on the `report_df` which has already been
    # filtered based on the `--include-validation` flag.
    total_real_videos = report_df[report_df['Method'].isin(REAL_METHODS)]['Processed_Unique_IDs'].sum()
    total_training_fakes = report_df[report_df['Method'].isin(TRAINING_FAKE_METHODS)]['Processed_Unique_IDs'].sum()
    total_validation_fakes = report_df[report_df['Method'].isin(VALIDATION_METHODS)]['Processed_Unique_IDs'].sum()

    console.print("\n[bold underline]Video Totals[/]")
    console.print(f"Total Real Videos: [bold green]{total_real_videos:,}[/]")
    console.print(f"Total Fake Videos (Training): [bold red]{total_training_fakes:,}[/]")

    if args.include_validation and total_validation_fakes > 0:
        console.print(f"Total Fake Videos (Validation): [bold yellow]{total_validation_fakes:,}[/]")
        total_fakes = total_training_fakes + total_validation_fakes
        console.print(f"Total Fake Videos (All): [bold red]{total_fakes:,}[/]")
    else:
        # If not including validation, the total is just the training fakes
        console.print(f"Total Fake Videos (All): [bold red]{total_training_fakes:,}[/]")

    console.print(
        "\n[italic]Note: 'Total Fake Videos' is a sum of unique IDs per method and may include duplicates if a source video is used for multiple fake methods.[/italic]"
    )

    # --- Interpretation Guide (MARKUP FIXED) ---
    console.print("\n[bold yellow]HOW TO INTERPRET THIS REPORT:[/]")
    console.print(
        "1. [bold]Frames Dropped %:[/bold] A high percentage indicates a problem processing a specific method (e.g., corrupt files, naming issues). A small percentage (~0.2%) is normal.")
    console.print("2. [bold]Source Videos vs. Processed IDs (The Key Insight):[/]")
    console.print("   - For a [bold]REAL[/] method like 'youtube-real', the two columns should be [bold]identical[/].")
    console.print(
        "   - For a [bold]FAKE[/] method like 'simswap', the 'Processed IDs' count should match its [bold]SOURCE's[/bold] 'Processed IDs' count (e.g., 'FaceForensics++').")
    console.print(
        "   - [bold red]If 'Source Videos' and 'Processed IDs' are the same for a FAKE method, your ID unification is NOT working.[/bold red]")


if __name__ == "__main__":
    main()
