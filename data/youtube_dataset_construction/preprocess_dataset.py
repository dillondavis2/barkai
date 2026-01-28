#!/usr/bin/env python3
"""
Dog Bark Dataset Preprocessing Script

This script preprocesses the dog bark audio dataset by:
1. Trimming silence from audio files
2. Extracting bark segments using energy-based detection
3. Splitting long files into shorter clips
4. Generating a manual review list for problematic files
5. Standardizing audio format (16kHz, mono, WAV)

Usage:
    # Preview what would be processed (dry run)
    python preprocess_dataset.py --data_dir youtube --output_dir youtube_cleaned --dry_run

    # Run full preprocessing
    python preprocess_dataset.py --data_dir youtube --output_dir youtube_cleaned

    # Only generate review list
    python preprocess_dataset.py --data_dir youtube --review_only

Author: Dillon Davis
"""

import argparse
import json
import os
import shutil
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Constants
TARGET_SR = 16000  # Target sample rate
MAX_DURATION = 30.0  # Maximum clip duration in seconds
MIN_DURATION = 1.0  # Minimum clip duration in seconds
MIN_BARK_DURATION = 0.5  # Minimum bark segment duration
SILENCE_THRESHOLD_DB = -40  # Silence threshold in dB
BARK_THRESHOLD_DB = -25  # Bark detection threshold in dB
HOP_LENGTH_MS = 10  # Hop length in milliseconds
FRAME_LENGTH_MS = 25  # Frame length in milliseconds


def load_audio(file_path: str, sr: int = TARGET_SR) -> Tuple[Optional[np.ndarray], int]:
    """Load audio file and resample."""
    try:
        import librosa
        audio, _ = librosa.load(file_path, sr=sr, mono=True)
        return audio, sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, sr


def save_audio(audio: np.ndarray, file_path: str, sr: int = TARGET_SR):
    """Save audio to WAV file."""
    import soundfile as sf
    sf.write(file_path, audio, sr)


def calculate_rms_db(audio: np.ndarray, sr: int,
                     frame_length_ms: int = FRAME_LENGTH_MS,
                     hop_length_ms: int = HOP_LENGTH_MS) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate RMS energy in dB for each frame."""
    import librosa

    frame_length = int(frame_length_ms * sr / 1000)
    hop_length = int(hop_length_ms * sr / 1000)

    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    # Time stamps for each frame
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    return rms_db, times


def detect_bark_segments(audio: np.ndarray, sr: int,
                        threshold_db: float = BARK_THRESHOLD_DB,
                        min_duration: float = MIN_BARK_DURATION,
                        merge_gap: float = 0.3) -> List[Tuple[float, float]]:
    """
    Detect bark segments based on energy threshold.

    Args:
        audio: Audio signal
        sr: Sample rate
        threshold_db: Energy threshold for bark detection (dB)
        min_duration: Minimum segment duration (seconds)
        merge_gap: Merge segments closer than this (seconds)

    Returns:
        List of (start_time, end_time) tuples for bark segments
    """
    rms_db, times = calculate_rms_db(audio, sr)

    # Find frames above threshold
    above_threshold = rms_db > threshold_db

    # Find segment boundaries
    segments = []
    in_segment = False
    start_time = 0

    for i, (is_above, t) in enumerate(zip(above_threshold, times)):
        if is_above and not in_segment:
            start_time = t
            in_segment = True
        elif not is_above and in_segment:
            end_time = t
            if end_time - start_time >= min_duration:
                segments.append((start_time, end_time))
            in_segment = False

    # Handle case where audio ends while in segment
    if in_segment:
        end_time = times[-1]
        if end_time - start_time >= min_duration:
            segments.append((start_time, end_time))

    # Merge nearby segments
    if len(segments) > 1:
        merged = [segments[0]]
        for start, end in segments[1:]:
            prev_start, prev_end = merged[-1]
            if start - prev_end < merge_gap:
                merged[-1] = (prev_start, end)
            else:
                merged.append((start, end))
        segments = merged

    return segments


def trim_silence(audio: np.ndarray, sr: int,
                threshold_db: float = SILENCE_THRESHOLD_DB,
                margin: float = 0.1) -> np.ndarray:
    """
    Trim leading and trailing silence from audio.

    Args:
        audio: Audio signal
        sr: Sample rate
        threshold_db: Silence threshold in dB
        margin: Extra margin to keep (seconds)

    Returns:
        Trimmed audio
    """
    import librosa

    # Use librosa's trim function
    trimmed, index = librosa.effects.trim(audio, top_db=abs(threshold_db))

    # Add small margin
    margin_samples = int(margin * sr)
    start = max(0, index[0] - margin_samples)
    end = min(len(audio), index[1] + margin_samples)

    return audio[start:end]


def split_into_clips(audio: np.ndarray, sr: int,
                    max_duration: float = MAX_DURATION,
                    overlap: float = 0.5) -> List[np.ndarray]:
    """
    Split long audio into shorter clips.

    Args:
        audio: Audio signal
        sr: Sample rate
        max_duration: Maximum clip duration (seconds)
        overlap: Overlap between clips (seconds)

    Returns:
        List of audio clips
    """
    duration = len(audio) / sr

    if duration <= max_duration:
        return [audio]

    clips = []
    max_samples = int(max_duration * sr)
    hop_samples = int((max_duration - overlap) * sr)

    start = 0
    while start < len(audio):
        end = min(start + max_samples, len(audio))
        clip = audio[start:end]

        # Only include if longer than minimum duration
        if len(clip) / sr >= MIN_DURATION:
            clips.append(clip)

        start += hop_samples

        # Avoid tiny final clips
        if len(audio) - start < MIN_DURATION * sr:
            break

    return clips


def extract_bark_clips(audio: np.ndarray, sr: int,
                      segments: List[Tuple[float, float]],
                      max_duration: float = MAX_DURATION,
                      context: float = 0.2) -> List[np.ndarray]:
    """
    Extract audio clips around detected bark segments.

    Args:
        audio: Audio signal
        sr: Sample rate
        segments: List of (start, end) bark segments
        max_duration: Maximum clip duration
        context: Extra context to include around barks (seconds)

    Returns:
        List of audio clips containing barks
    """
    clips = []

    for start, end in segments:
        # Add context
        clip_start = max(0, start - context)
        clip_end = min(len(audio) / sr, end + context)

        # Extract samples
        start_sample = int(clip_start * sr)
        end_sample = int(clip_end * sr)
        clip = audio[start_sample:end_sample]

        # Split if too long
        if len(clip) / sr > max_duration:
            sub_clips = split_into_clips(clip, sr, max_duration)
            clips.extend(sub_clips)
        elif len(clip) / sr >= MIN_DURATION:
            clips.append(clip)

    return clips


def process_audio_file(input_path: str, output_dir: Path, breed: str,
                      file_index: int, strategy: str = "auto") -> Dict:
    """
    Process a single audio file.

    Args:
        input_path: Path to input audio file
        output_dir: Output directory
        breed: Breed name
        file_index: Index for output filename
        strategy: Processing strategy ("auto", "trim", "extract_barks", "split")

    Returns:
        Dict with processing results
    """
    result = {
        "input_file": input_path,
        "breed": breed,
        "strategy": strategy,
        "output_files": [],
        "status": "success",
        "error": None,
    }

    # Load audio
    audio, sr = load_audio(input_path)
    if audio is None:
        result["status"] = "error"
        result["error"] = "Failed to load audio"
        return result

    original_duration = len(audio) / sr
    result["original_duration"] = original_duration

    # Determine strategy if auto
    if strategy == "auto":
        if original_duration > 120:  # Very long - extract barks
            strategy = "extract_barks"
        elif original_duration > MAX_DURATION:  # Moderately long - split
            strategy = "split"
        else:  # Short enough - just trim silence
            strategy = "trim"

    result["strategy"] = strategy

    # Process based on strategy
    clips = []

    if strategy == "extract_barks":
        # Detect bark segments
        segments = detect_bark_segments(audio, sr)
        if segments:
            clips = extract_bark_clips(audio, sr, segments)
        else:
            # Fallback to splitting if no barks detected
            clips = split_into_clips(trim_silence(audio, sr), sr)

    elif strategy == "split":
        # Trim silence then split
        trimmed = trim_silence(audio, sr)
        clips = split_into_clips(trimmed, sr)

    elif strategy == "trim":
        # Just trim silence
        trimmed = trim_silence(audio, sr)
        if len(trimmed) / sr >= MIN_DURATION:
            clips = [trimmed]

    # Save clips
    breed_dir = output_dir / breed
    breed_dir.mkdir(parents=True, exist_ok=True)

    for i, clip in enumerate(clips):
        if len(clips) == 1:
            output_filename = f"{breed}_{file_index:04d}.wav"
        else:
            output_filename = f"{breed}_{file_index:04d}_{i:02d}.wav"

        output_path = breed_dir / output_filename
        save_audio(clip, str(output_path), sr)
        result["output_files"].append({
            "path": str(output_path),
            "duration": len(clip) / sr,
        })

    result["num_clips"] = len(clips)
    result["total_output_duration"] = sum(len(c) / sr for c in clips) if clips else 0

    return result


def generate_review_list(data_dir: Path, output_path: Path,
                        duration_threshold: float = 120,
                        silence_threshold: float = 0.8,
                        energy_threshold: float = 0.001) -> pd.DataFrame:
    """
    Generate a list of files that need manual review.
    """
    import librosa

    print("\n" + "=" * 60)
    print("GENERATING MANUAL REVIEW LIST")
    print("=" * 60)

    review_items = []

    all_files = []
    for breed_dir in data_dir.iterdir():
        if breed_dir.is_dir() and breed_dir.name != "validation_report":
            for ext in ["*.wav", "*.mp3", "*.m4a"]:
                all_files.extend([(breed_dir.name, f) for f in breed_dir.glob(ext)])

    print(f"Scanning {len(all_files)} files...")

    for breed, file_path in tqdm(all_files, desc="Analyzing"):
        issues = []

        try:
            # Get duration
            duration = librosa.get_duration(path=str(file_path))

            if duration > duration_threshold:
                issues.append(f"Long duration: {duration:.1f}s")

            if duration < 3:
                issues.append(f"Short duration: {duration:.1f}s")

            # Load and check quality (sample for speed)
            if duration < 300:  # Only fully analyze files < 5 min
                audio, sr = load_audio(str(file_path))
                if audio is not None:
                    rms = np.sqrt(np.mean(audio ** 2))
                    if rms < energy_threshold:
                        issues.append(f"Low energy: {rms:.4f}")

                    # Check silence ratio
                    rms_db, _ = calculate_rms_db(audio, sr)
                    silence_ratio = np.mean(rms_db < SILENCE_THRESHOLD_DB)
                    if silence_ratio > silence_threshold:
                        issues.append(f"High silence: {silence_ratio:.1%}")
            else:
                issues.append("Too long to fully analyze")

            if issues:
                review_items.append({
                    "breed": breed,
                    "file": file_path.name,
                    "path": str(file_path),
                    "duration": duration,
                    "issues": "; ".join(issues),
                    "priority": "high" if duration > 300 or len(issues) > 1 else "medium",
                })

        except Exception as e:
            review_items.append({
                "breed": breed,
                "file": file_path.name,
                "path": str(file_path),
                "duration": -1,
                "issues": f"Error: {str(e)}",
                "priority": "high",
            })

    df = pd.DataFrame(review_items)
    df = df.sort_values(["priority", "breed", "duration"], ascending=[True, True, False])

    df.to_csv(output_path, index=False)

    print(f"\nFiles needing review: {len(df)}")
    print(f"  High priority: {len(df[df['priority'] == 'high'])}")
    print(f"  Medium priority: {len(df[df['priority'] == 'medium'])}")
    print(f"\nReview list saved to: {output_path}")

    # Print summary by issue type
    print("\nIssue summary:")
    issue_counts = defaultdict(int)
    for issues in df['issues']:
        for issue in issues.split("; "):
            issue_type = issue.split(":")[0]
            issue_counts[issue_type] += 1

    for issue_type, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
        print(f"  {issue_type}: {count}")

    return df


def preprocess_dataset(data_dir: Path, output_dir: Path,
                      dry_run: bool = False,
                      strategy: str = "auto") -> Dict:
    """
    Preprocess the entire dataset.
    """
    print("\n" + "=" * 60)
    print("PREPROCESSING DATASET")
    print("=" * 60)
    print(f"Input directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Strategy: {strategy}")
    print(f"Dry run: {dry_run}")

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all files
    all_files = []
    for breed_dir in sorted(data_dir.iterdir()):
        if breed_dir.is_dir() and breed_dir.name != "validation_report":
            breed_files = []
            for ext in ["*.wav", "*.mp3", "*.m4a"]:
                breed_files.extend(list(breed_dir.glob(ext)))
            breed_files.sort()
            for i, f in enumerate(breed_files):
                all_files.append((breed_dir.name, f, i))

    print(f"\nTotal files to process: {len(all_files)}")

    results = []
    stats = {
        "total_input_files": len(all_files),
        "total_output_files": 0,
        "total_input_duration": 0,
        "total_output_duration": 0,
        "errors": 0,
        "by_strategy": defaultdict(int),
        "by_breed": defaultdict(lambda: {"input": 0, "output": 0}),
    }

    for breed, file_path, idx in tqdm(all_files, desc="Processing"):
        if dry_run:
            # Just analyze what would happen
            try:
                import librosa
                duration = librosa.get_duration(path=str(file_path))
                stats["total_input_duration"] += duration
                stats["by_breed"][breed]["input"] += 1

                if strategy == "auto":
                    if duration > 120:
                        strat = "extract_barks"
                    elif duration > MAX_DURATION:
                        strat = "split"
                    else:
                        strat = "trim"
                else:
                    strat = strategy
                stats["by_strategy"][strat] += 1
            except:
                stats["errors"] += 1
        else:
            result = process_audio_file(
                str(file_path), output_dir, breed, idx, strategy
            )
            results.append(result)

            if result["status"] == "success":
                stats["total_output_files"] += result["num_clips"]
                stats["total_input_duration"] += result.get("original_duration", 0)
                stats["total_output_duration"] += result.get("total_output_duration", 0)
                stats["by_strategy"][result["strategy"]] += 1
                stats["by_breed"][breed]["input"] += 1
                stats["by_breed"][breed]["output"] += result["num_clips"]
            else:
                stats["errors"] += 1

    # Print summary
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"Input files: {stats['total_input_files']}")
    print(f"Input duration: {stats['total_input_duration'] / 3600:.2f} hours")

    if not dry_run:
        print(f"Output files: {stats['total_output_files']}")
        print(f"Output duration: {stats['total_output_duration'] / 3600:.2f} hours")
        print(f"Compression ratio: {stats['total_output_duration'] / max(stats['total_input_duration'], 1):.1%}")

    print(f"Errors: {stats['errors']}")

    print("\nBy strategy:")
    for strat, count in sorted(stats["by_strategy"].items()):
        print(f"  {strat}: {count}")

    if not dry_run:
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_dir / "preprocessing_results.csv", index=False)

        with open(output_dir / "preprocessing_stats.json", "w") as f:
            # Convert defaultdicts to regular dicts for JSON
            stats_json = {
                k: dict(v) if isinstance(v, defaultdict) else v
                for k, v in stats.items()
            }
            json.dump(stats_json, f, indent=2)

        print(f"\nResults saved to: {output_dir}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Preprocess dog bark audio dataset")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Path to input dataset directory")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for processed files")
    parser.add_argument("--strategy", type=str, default="auto",
                       choices=["auto", "trim", "extract_barks", "split"],
                       help="Processing strategy")
    parser.add_argument("--dry_run", action="store_true",
                       help="Preview what would be processed without making changes")
    parser.add_argument("--review_only", action="store_true",
                       help="Only generate the manual review list")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return

    if args.review_only:
        review_path = data_dir / "manual_review_list.csv"
        generate_review_list(data_dir, review_path)
    else:
        output_dir = Path(args.output_dir) if args.output_dir else data_dir.parent / f"{data_dir.name}_cleaned"

        # Generate review list first
        review_path = data_dir / "manual_review_list.csv"
        generate_review_list(data_dir, review_path)

        # Then preprocess
        preprocess_dataset(data_dir, output_dir, args.dry_run, args.strategy)


if __name__ == "__main__":
    main()
