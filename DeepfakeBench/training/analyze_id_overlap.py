#!/usr/bin/env python3
"""
analyze_id_overlap.py

Analyzes the overlap of 'video_id's between different manipulation methods
in the processed manifest.
"""
import pandas as pd
from rich.console import Console
from rich.table import Table

MANIFEST_PATH = "frame_properties.parquet"

# --- Configuration ---
REFERENCE_METHOD = "FaceForensics++"
METHODS_TO_COMPARE = [
    "simswap", "mobileswap", "faceswap", "inswap", "blendface", "fsgan", "uniface",
    "pirender", "facevid2vid", "lia", "fomm", "MRAA", "wav2lip", "mcnet", "danet",
    "ddim"
]


def jaccard_similarity(set1, set2):
    """Calculates the Jaccard similarity between two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0.0


def main():
    """Runs the full Jaccard similarity analysis."""
    console = Console()
    console.print(f"[bold cyan]Loading manifest:[/] '{MANIFEST_PATH}'")
    try:
        df = pd.read_parquet(MANIFEST_PATH)
    except FileNotFoundError:
        console.print(f"[bold red]Error:[/] Manifest not found.")
        return

    console.print(f"\nAnalyzing ID overlap against reference method: [bold magenta]'{REFERENCE_METHOD}'[/]\n")

    # --- THIS NOW USES THE UNIFIED 'video_id' COLUMN ---
    all_methods_in_df = df['method'].unique()
    if REFERENCE_METHOD not in all_methods_in_df:
        console.print(f"[bold red]Error:[/] Reference method '{REFERENCE_METHOD}' not found in the manifest.")
        return
    reference_ids = set(df[df['method'] == REFERENCE_METHOD]['video_id'])

    table = Table(title="Source Video ID Overlap Analysis")
    table.add_column("Method Compared", style="cyan")
    table.add_column(f"vs. {REFERENCE_METHOD}", style="magenta", justify="center")
    table.add_column("Reference IDs", justify="right", style="green")
    table.add_column("Method IDs", justify="right", style="green")
    table.add_column("Shared IDs\n(Intersection)", justify="right", style="yellow")
    table.add_column("Jaccard Sim.\n(Overlap Score)", justify="right", style="bold red")

    sorted_methods_to_compare = sorted([m for m in METHODS_TO_COMPARE if m in all_methods_in_df])

    for method in sorted_methods_to_compare:
        method_ids = set(df[df['method'] == method]['video_id'])
        num_reference_ids = len(reference_ids)
        num_method_ids = len(method_ids)
        num_intersection = len(reference_ids.intersection(method_ids))
        similarity = jaccard_similarity(reference_ids, method_ids)
        sim_style = "bold green" if similarity > 0.9 else "bold yellow" if similarity > 0.5 else "bold red"
        table.add_row(
            method, "<->", f"{num_reference_ids:,}", f"{num_method_ids:,}",
            f"{num_intersection:,}", f"[{sim_style}]{similarity:.3f}[/]"
        )

    console.print(table)


def inspect_id_mapping(manifest_path: str):
    """
    Loads the manifest and shows a side-by-side comparison of the original
    and the newly computed unified video IDs.
    """
    console = Console()
    console.print(f"🔎 [bold cyan]Inspecting ID mapping in:[/] '{manifest_path}'")
    try:
        df = pd.read_parquet(manifest_path)
        required_cols = ['original_video_id', 'video_id']
        if not all(col in df.columns for col in required_cols):
            console.print(f"❌ [bold red]ERROR:[/] Manifest is missing required columns. Expected: {required_cols}")
            return
    except FileNotFoundError:
        console.print(f"❌ [bold red]ERROR:[/] Manifest file not found at '{manifest_path}'.")
        return

    console.print("\n--- ✅ Verifying 'wav2lip' (Fake) ---")
    console.print("The unified 'video_id' should be the part [bold]before[/] the first underscore.")
    wav2lip_df = df[df['method'] == 'wav2lip'].drop_duplicates(subset=['original_video_id']).head(5)
    table_wav2lip = Table(show_header=True, header_style="bold magenta")
    table_wav2lip.add_column("Preserved 'original_video_id'", style="dim")
    table_wav2lip.add_column("Unified 'video_id'", style="bold green")
    for _, row in wav2lip_df.iterrows():
        table_wav2lip.add_row(row['original_video_id'], str(row['video_id']))
    console.print(table_wav2lip)

    console.print("\n--- ✅ Verifying 'blendface' (Fake) ---")
    console.print("The unified 'video_id' should be the part [bold]after[/] the underscore.")
    blendface_df = df[df['method'] == 'blendface'].drop_duplicates(subset=['original_video_id']).head(5)
    table_bf = Table(show_header=True, header_style="bold magenta")
    table_bf.add_column("Preserved 'original_video_id'", style="dim")
    table_bf.add_column("Unified 'video_id'", style="bold green")
    for _, row in blendface_df.iterrows():
        table_bf.add_row(row['original_video_id'], str(row['video_id']))
    console.print(table_bf)


if __name__ == "__main__":
    # --- STEP 1: Run this inspection first! ---
    inspect_id_mapping(manifest_path=MANIFEST_PATH)

    # --- STEP 2: Once inspection looks good, comment out the line above and uncomment the line below ---
    main()