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
    python diagnose_manifests.py
"""
import json
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table


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
    # This fixes the KeyError.
    analysis = df.groupby('method').agg(
        Processed_Frames=('video_id', 'size'),
        Processed_Unique_IDs=('video_id', 'nunique')
    ).reset_index()
    analysis.rename(columns={'method': 'Method'}, inplace=True)

    return analysis.sort_values('Method')


def main():
    """Main execution function."""
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
    console.print("\n[bold yellow]HOW TO INTERPRET THIS REPORT:[/]")
    console.print(
        "1. [bold]Frames Dropped %:[/bold] A high percentage indicates a problem processing a specific method (e.g., corrupt files, naming issues). A small percentage (~0.2%) is normal.")
    console.print("2. [bold]Source Videos vs. Processed IDs (The Key Insight):[/]")
    console.print("   - For a [bold]REAL[/] method like 'youtube-real', the two columns should be [bold]identical[/].")
    console.print(
        "   - For a [bold]FAKE[/] method like 'simswap', the 'Processed IDs' count should match its [bold]SOURCE's[/bold] 'Processed IDs' count (e.g., 'FaceForensics++').")
    console.print(
        "   - [bold red]If 'Source Videos' and 'Processed IDs' are the same for a FAKE method, your ID unification is NOT working.[/bold]")


if __name__ == "__main__":
    main()
