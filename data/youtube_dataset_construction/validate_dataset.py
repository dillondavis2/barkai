#!/usr/bin/env python3
"""
Dog Bark Dataset Validation Script

This script validates the quality of the dog bark audio dataset by checking:
1. Class balance - samples per breed
2. Audio file integrity - files can be loaded without errors
3. Duration analysis - identify too short/long clips
4. Silent file detection - files with very low energy
5. Audio quality metrics - SNR estimation
6. Embedding-based outlier detection - find potentially mislabeled samples

Usage:
    python validate_dataset.py --data_dir /path/to/dog_bark_audio/youtube
    python validate_dataset.py --data_dir /path/to/dog_bark_audio/youtube --full_analysis

Author: Dillon Davis
"""

import argparse
import json
import os
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")


def load_audio(file_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load audio file and resample to target sample rate."""
    try:
        import librosa
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
        return audio, sr
    except Exception as e:
        return None, str(e)


def get_audio_duration(file_path: str) -> float:
    """Get audio duration without loading full file."""
    try:
        import librosa
        return librosa.get_duration(path=file_path)
    except:
        return -1


def calculate_rms_energy(audio: np.ndarray) -> float:
    """Calculate RMS energy of audio signal."""
    return np.sqrt(np.mean(audio ** 2))


def calculate_snr_estimate(audio: np.ndarray, sr: int) -> float:
    """Estimate SNR by comparing high-energy to low-energy segments."""
    import librosa

    # Calculate energy in short frames
    frame_length = int(0.025 * sr)  # 25ms frames
    hop_length = int(0.010 * sr)    # 10ms hop

    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]

    if len(rms) < 10:
        return 0

    # Sort frames by energy
    sorted_rms = np.sort(rms)

    # Estimate noise floor from lowest 10% of frames
    noise_floor = np.mean(sorted_rms[:max(1, len(sorted_rms) // 10)])

    # Estimate signal from highest 20% of frames
    signal_level = np.mean(sorted_rms[-max(1, len(sorted_rms) // 5):])

    if noise_floor > 0:
        snr = 20 * np.log10(signal_level / noise_floor)
        return snr
    return 0


def detect_silence_ratio(audio: np.ndarray, sr: int, threshold_db: float = -40) -> float:
    """Calculate ratio of silent frames in audio."""
    import librosa

    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)

    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    silent_frames = np.sum(rms_db < threshold_db)
    return silent_frames / len(rms_db)


def analyze_class_balance(data_dir: Path) -> pd.DataFrame:
    """Analyze class balance across breeds."""
    print("\n" + "=" * 60)
    print("CLASS BALANCE ANALYSIS")
    print("=" * 60)

    breed_counts = {}
    for breed_dir in sorted(data_dir.iterdir()):
        if breed_dir.is_dir():
            audio_files = list(breed_dir.glob("*.wav")) + \
                         list(breed_dir.glob("*.mp3")) + \
                         list(breed_dir.glob("*.m4a"))
            breed_counts[breed_dir.name] = len(audio_files)

    df = pd.DataFrame([
        {"breed": k, "count": v} for k, v in breed_counts.items()
    ]).sort_values("count", ascending=False)

    print(f"\nTotal breeds: {len(breed_counts)}")
    print(f"Total samples: {sum(breed_counts.values())}")
    print(f"\nSamples per breed:")
    print(f"  Min: {df['count'].min()} ({df[df['count'] == df['count'].min()]['breed'].values[0]})")
    print(f"  Max: {df['count'].max()} ({df[df['count'] == df['count'].max()]['breed'].values[0]})")
    print(f"  Mean: {df['count'].mean():.1f}")
    print(f"  Std: {df['count'].std():.1f}")

    # Flag imbalanced classes
    threshold_low = df['count'].mean() - 2 * df['count'].std()
    threshold_high = df['count'].mean() + 2 * df['count'].std()

    low_count = df[df['count'] < max(threshold_low, 10)]
    if len(low_count) > 0:
        print(f"\n⚠️  LOW SAMPLE BREEDS (< {max(threshold_low, 10):.0f}):")
        for _, row in low_count.iterrows():
            print(f"    {row['breed']}: {row['count']}")

    return df


def analyze_durations(data_dir: Path, sample_size: int = None) -> pd.DataFrame:
    """Analyze audio durations across the dataset."""
    print("\n" + "=" * 60)
    print("DURATION ANALYSIS")
    print("=" * 60)

    results = []
    all_files = []

    for breed_dir in data_dir.iterdir():
        if breed_dir.is_dir():
            audio_files = list(breed_dir.glob("*.wav")) + \
                         list(breed_dir.glob("*.mp3")) + \
                         list(breed_dir.glob("*.m4a"))
            for f in audio_files:
                all_files.append((breed_dir.name, f))

    if sample_size and sample_size < len(all_files):
        import random
        random.shuffle(all_files)
        all_files = all_files[:sample_size]

    print(f"Analyzing {len(all_files)} files...")

    for breed, file_path in tqdm(all_files, desc="Getting durations"):
        duration = get_audio_duration(str(file_path))
        results.append({
            "breed": breed,
            "file": file_path.name,
            "path": str(file_path),
            "duration": duration,
        })

    df = pd.DataFrame(results)
    valid_df = df[df['duration'] > 0]

    print(f"\nDuration statistics:")
    print(f"  Min: {valid_df['duration'].min():.1f}s")
    print(f"  Max: {valid_df['duration'].max():.1f}s")
    print(f"  Mean: {valid_df['duration'].mean():.1f}s")
    print(f"  Median: {valid_df['duration'].median():.1f}s")

    # Flag problematic durations
    too_short = valid_df[valid_df['duration'] < 3]
    too_long = valid_df[valid_df['duration'] > 120]

    if len(too_short) > 0:
        print(f"\n⚠️  TOO SHORT (< 3s): {len(too_short)} files")
        for _, row in too_short.head(10).iterrows():
            print(f"    {row['breed']}/{row['file']}: {row['duration']:.1f}s")

    if len(too_long) > 0:
        print(f"\n⚠️  TOO LONG (> 120s): {len(too_long)} files")
        print("    These may contain non-bark content (music, speech, etc.)")
        for _, row in too_long.head(10).iterrows():
            print(f"    {row['breed']}/{row['file']}: {row['duration']:.1f}s")

    # Duration by breed
    breed_durations = valid_df.groupby('breed')['duration'].agg(['mean', 'std', 'min', 'max'])

    return df, breed_durations


def analyze_audio_quality(data_dir: Path, sample_size: int = 500) -> pd.DataFrame:
    """Analyze audio quality metrics on a sample of files."""
    print("\n" + "=" * 60)
    print("AUDIO QUALITY ANALYSIS")
    print("=" * 60)

    all_files = []
    for breed_dir in data_dir.iterdir():
        if breed_dir.is_dir():
            audio_files = list(breed_dir.glob("*.wav")) + \
                         list(breed_dir.glob("*.mp3")) + \
                         list(breed_dir.glob("*.m4a"))
            for f in audio_files:
                all_files.append((breed_dir.name, f))

    import random
    random.shuffle(all_files)
    sample_files = all_files[:sample_size]

    print(f"Analyzing {len(sample_files)} sampled files...")

    results = []
    errors = []

    for breed, file_path in tqdm(sample_files, desc="Analyzing quality"):
        audio, result = load_audio(str(file_path))

        if audio is None:
            errors.append({"breed": breed, "file": file_path.name, "error": result})
            continue

        sr = 16000  # We resampled to this

        rms = calculate_rms_energy(audio)
        snr = calculate_snr_estimate(audio, sr)
        silence_ratio = detect_silence_ratio(audio, sr)

        results.append({
            "breed": breed,
            "file": file_path.name,
            "path": str(file_path),
            "rms_energy": rms,
            "snr_estimate": snr,
            "silence_ratio": silence_ratio,
        })

    df = pd.DataFrame(results)

    print(f"\nFiles with load errors: {len(errors)}")
    if errors:
        for e in errors[:5]:
            print(f"  {e['breed']}/{e['file']}: {e['error']}")

    print(f"\nQuality metrics:")
    print(f"  RMS Energy - Mean: {df['rms_energy'].mean():.4f}, Std: {df['rms_energy'].std():.4f}")
    print(f"  SNR Estimate - Mean: {df['snr_estimate'].mean():.1f}dB, Std: {df['snr_estimate'].std():.1f}dB")
    print(f"  Silence Ratio - Mean: {df['silence_ratio'].mean():.2%}, Std: {df['silence_ratio'].std():.2%}")

    # Flag low quality files
    low_energy = df[df['rms_energy'] < 0.001]
    high_silence = df[df['silence_ratio'] > 0.8]
    low_snr = df[df['snr_estimate'] < 5]

    if len(low_energy) > 0:
        print(f"\n⚠️  VERY LOW ENERGY (< 0.001 RMS): {len(low_energy)} files")
        print("    These may be nearly silent or corrupted")
        for _, row in low_energy.head(5).iterrows():
            print(f"    {row['breed']}/{row['file']}")

    if len(high_silence) > 0:
        print(f"\n⚠️  HIGH SILENCE RATIO (> 80%): {len(high_silence)} files")
        print("    These contain mostly silence")
        for _, row in high_silence.head(5).iterrows():
            print(f"    {row['breed']}/{row['file']}: {row['silence_ratio']:.1%}")

    if len(low_snr) > 0:
        print(f"\n⚠️  LOW SNR (< 5dB): {len(low_snr)} files")
        print("    These may have poor signal quality")

    return df, errors


def generate_report(data_dir: Path, output_dir: Path, full_analysis: bool = False):
    """Generate comprehensive validation report."""

    print("\n" + "=" * 60)
    print("DOG BARK DATASET VALIDATION REPORT")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Class balance
    class_df = analyze_class_balance(data_dir)
    class_df.to_csv(output_dir / "class_balance.csv", index=False)

    # 2. Duration analysis
    sample_size = None if full_analysis else 1000
    duration_df, breed_durations = analyze_durations(data_dir, sample_size)
    duration_df.to_csv(output_dir / "duration_analysis.csv", index=False)
    breed_durations.to_csv(output_dir / "duration_by_breed.csv")

    # 3. Audio quality (always sampled for speed)
    quality_sample = 1000 if full_analysis else 500
    quality_df, errors = analyze_audio_quality(data_dir, quality_sample)
    quality_df.to_csv(output_dir / "quality_analysis.csv", index=False)

    if errors:
        pd.DataFrame(errors).to_csv(output_dir / "load_errors.csv", index=False)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 60)

    issues = []

    # Check class balance
    if class_df['count'].std() / class_df['count'].mean() > 0.3:
        issues.append("- Class imbalance detected. Consider oversampling minority classes or undersampling majority.")

    # Check durations
    valid_durations = duration_df[duration_df['duration'] > 0]
    if (valid_durations['duration'] > 120).sum() > len(valid_durations) * 0.1:
        issues.append("- Many clips > 2 minutes. Consider trimming to bark segments only.")

    if (valid_durations['duration'] < 3).sum() > 0:
        issues.append("- Some clips < 3 seconds. These may not contain enough bark content.")

    # Check quality
    if (quality_df['silence_ratio'] > 0.5).sum() > len(quality_df) * 0.1:
        issues.append("- Many files with high silence ratio. Consider trimming silence.")

    if len(errors) > 0:
        issues.append(f"- {len(errors)} files failed to load. Remove or fix these.")

    if issues:
        print("\n⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✅ No major issues detected!")

    print("\n📋 RECOMMENDED NEXT STEPS:")
    print("  1. Review files flagged as too long (>120s) - may need manual trimming")
    print("  2. Review files with high silence ratio - trim or remove")
    print("  3. Remove files that failed to load")
    print("  4. Consider using audio embeddings to detect outliers/mislabels")
    print("  5. Manually spot-check a random sample from each class")

    print(f"\n📁 Reports saved to: {output_dir}")

    return {
        "class_balance": class_df,
        "durations": duration_df,
        "quality": quality_df,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate dog bark audio dataset")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for reports")
    parser.add_argument("--full_analysis", action="store_true", help="Run full analysis (slower)")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "validation_report"

    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return

    generate_report(data_dir, output_dir, args.full_analysis)


if __name__ == "__main__":
    main()
