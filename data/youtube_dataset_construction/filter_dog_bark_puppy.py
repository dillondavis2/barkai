#!/usr/bin/env python3
"""
Filter audio files from youtube_cleaned that have "Dog", "Bark", or "Puppy" in their top-5 predictions.

Usage:
    python filter_dog_bark_puppy.py --results <results_json> --source <youtube_cleaned_dir> --output <output_dir>
"""

import argparse
import json
import shutil
from pathlib import Path


def has_dog_bark_puppy_in_top5(top_predictions: list[dict]) -> bool:
    """Check if 'Dog', 'Bark', or 'Puppy' appears in the top-5 predictions."""
    if not top_predictions:
        return False
    keywords = {"dog", "bark", "puppy"}
    for pred in top_predictions[:5]:
        class_name = pred.get("class", "").lower()
        if any(kw in class_name for kw in keywords):
            return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Filter audio files with 'Dog', 'Bark', or 'Puppy' in top-5 predictions"
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path.home() / "repos/barkai/youtube_filter_results.json",
        help="Path to the AST filter results JSON file",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path.home() / "airlab/dog_bark_audio/youtube_cleaned",
        help="Path to the youtube_cleaned source directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.home() / "airlab/dog_bark_audio/youtube_filtered",
        help="Path to the output directory",
    )
    args = parser.parse_args()

    # Load results
    print(f"Loading results from {args.results}")
    with open(args.results) as f:
        results = json.load(f)

    print(f"Total files in results: {len(results)}")

    # Filter for Dog/Bark/Puppy in top-5
    filtered = []
    for entry in results:
        if has_dog_bark_puppy_in_top5(entry.get("top_predictions", [])):
            filtered.append(entry)

    print(f"Files with 'Dog', 'Bark', or 'Puppy' in top-5 predictions: {len(filtered)}")

    # Copy filtered files to output directory, preserving breed structure
    args.output.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0
    breeds_seen = set()

    for entry in filtered:
        file_path = entry["file_path"]
        # file_path is like "youtube_cleaned/breed/filename.wav"
        parts = Path(file_path).parts
        if len(parts) >= 3:
            breed = parts[1]
            filename = parts[2]
        else:
            # Fallback: extract breed from path
            breed = Path(file_path).parent.name
            filename = Path(file_path).name

        breeds_seen.add(breed)

        # Source file path
        source_file = args.source / breed / filename

        # Destination directory and file
        dest_dir = args.output / breed
        dest_file = dest_dir / filename

        if not source_file.exists():
            print(f"Warning: Source file not found: {source_file}")
            skipped += 1
            continue

        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, dest_file)
        copied += 1

    print(f"\nSummary:")
    print(f"  Copied: {copied} files")
    print(f"  Skipped (not found): {skipped} files")
    print(f"  Breeds: {len(breeds_seen)}")
    print(f"  Output directory: {args.output}")

    # Print breakdown by breed
    breed_counts = {}
    for entry in filtered:
        file_path = entry["file_path"]
        parts = Path(file_path).parts
        breed = parts[1] if len(parts) >= 3 else Path(file_path).parent.name
        breed_counts[breed] = breed_counts.get(breed, 0) + 1

    print(f"\nBreakdown by breed:")
    for breed, count in sorted(breed_counts.items(), key=lambda x: -x[1]):
        print(f"  {breed}: {count}")


if __name__ == "__main__":
    main()
