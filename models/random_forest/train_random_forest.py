import os
import argparse
import numpy as np
import librosa
import joblib
import soundfile as sf
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
from scipy.ndimage import binary_dilation, binary_erosion
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm


# =============================================================================
# Bark Isolation Functions (Energy-Based Detection)
# =============================================================================

def isolate_barks_energy(
    audio_path: str,
    sample_rate: int = 16000,
    min_bark_duration: float = 0.3,
    max_bark_duration: float = 1.5,
    energy_threshold_db: float = -20,
    min_silence_between: float = 0.05,
) -> List[Tuple[float, float]]:
    """
    Isolate individual bark events using energy-based detection.

    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate
        min_bark_duration: Minimum bark duration in seconds
        max_bark_duration: Maximum bark duration in seconds
        energy_threshold_db: Energy threshold relative to max (dB)
        min_silence_between: Minimum silence between barks (seconds)

    Returns:
        List of (start_time, end_time) tuples for each detected bark
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=sample_rate)

        if len(y) == 0:
            return []

        # Compute short-time energy (RMS in small windows)
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)    # 10ms hop

        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

        if len(rms) == 0 or np.max(rms) == 0:
            return []

        # Convert to dB and normalize
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)

        # Threshold to find active regions
        is_active = rms_db > energy_threshold_db

        # Morphological operations to clean up detections
        # Dilate to merge nearby segments, erode to remove tiny spikes
        is_active = binary_dilation(is_active, iterations=3)
        is_active = binary_erosion(is_active, iterations=2)

        # Find contiguous regions
        bark_regions = []
        in_bark = False
        start_frame = 0

        for i, active in enumerate(is_active):
            if active and not in_bark:
                start_frame = i
                in_bark = True
            elif not active and in_bark:
                end_frame = i
                in_bark = False

                # Convert frames to time
                start_time = start_frame * hop_length / sr
                end_time = end_frame * hop_length / sr
                duration = end_time - start_time

                # Filter by duration
                if min_bark_duration <= duration <= max_bark_duration:
                    bark_regions.append((start_time, end_time))

        # Handle case where audio ends during a bark
        if in_bark:
            end_time = len(is_active) * hop_length / sr
            duration = end_time - (start_frame * hop_length / sr)
            if min_bark_duration <= duration <= max_bark_duration:
                bark_regions.append((start_frame * hop_length / sr, end_time))

        # Merge regions that are very close together
        bark_regions = merge_bark_regions(bark_regions, min_silence_between)

        return bark_regions

    except Exception as e:
        print(f"Error in bark isolation for {audio_path}: {e}")
        return []


def merge_bark_regions(
    regions: List[Tuple[float, float]], gap_threshold: float = 0.05
) -> List[Tuple[float, float]]:
    """Merge bark regions that are very close together."""
    if not regions:
        return []

    sorted_regions = sorted(regions)
    merged = [sorted_regions[0]]

    for start, end in sorted_regions[1:]:
        prev_start, prev_end = merged[-1]

        if start <= prev_end + gap_threshold:
            # Merge with previous
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged


def extract_bark_segment(
    audio: np.ndarray,
    start_time: float,
    end_time: float,
    sample_rate: int = 16000,
    padding: float = 0.05,
) -> Optional[np.ndarray]:
    """
    Extract a single bark segment from loaded audio.

    Args:
        audio: Audio samples as numpy array
        start_time: Start time of bark in seconds
        end_time: End time of bark in seconds
        sample_rate: Sample rate of audio
        padding: Padding around the bark in seconds

    Returns:
        Numpy array of audio samples, or None if extraction fails
    """
    try:
        # Add padding around the bark
        start_sample = max(0, int((start_time - padding) * sample_rate))
        end_sample = min(len(audio), int((end_time + padding) * sample_rate))

        segment = audio[start_sample:end_sample]

        if len(segment) == 0:
            return None

        return segment

    except Exception as e:
        print(f"Error extracting bark segment: {e}")
        return None


# =============================================================================
# Feature Extraction Functions
# =============================================================================

def extract_breed_from_filename(filename):
    """
    Extract breed name from WAV filename.
    Format: {integer}_{breed_name}_{gender}_dog_{integer}.wav
    Example: 58341_shibainu_F_dog_108.wav -> shibainu
    """
    # Remove .wav extension
    name = filename.replace('.wav', '')
    parts = name.split('_')

    # Format: {integer}_{breed_name}_{gender}_dog_{integer}
    # Parts: [0]=integer, [1]=breed_name, [2]=gender, [3]='dog', [4]=integer
    if len(parts) >= 5:
        return parts[1].lower()
    return None


def extract_features_from_audio(y: np.ndarray, sr: int = 16000) -> Optional[np.ndarray]:
    """
    Extracts 40 features from audio samples:
    - 20 MFCCs (Mean across time)
    - 20 MFCCs (Std Dev across time)

    Args:
        y: Audio samples as numpy array
        sr: Sample rate

    Returns:
        40-dimensional feature vector, or None if extraction fails
    """
    try:
        if len(y) == 0:
            return None

        # Remove silence (optional, but recommended)
        y, _ = librosa.effects.trim(y)

        if len(y) == 0:
            return None

        # Extract MFCCs (n_mfcc=20 is standard)
        # Output shape: (20, time_steps)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

        # Collapse Time: Take Mean and Std of each MFCC band
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        # Concatenate to get a 40-dim vector
        return np.concatenate([mfcc_mean, mfcc_std])

    except Exception as e:
        return None


def extract_features(file_path):
    """
    Extracts 40 features per audio clip:
    - 20 MFCCs (Mean across time)
    - 20 MFCCs (Std Dev across time)
    """
    try:
        # Load audio (resample to 16k for speed)
        y, sr = librosa.load(file_path, sr=16000)
        return extract_features_from_audio(y, sr)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def extract_features_with_bark_isolation(
    file_path: str,
    min_bark_duration: float = 0.3,
    max_bark_duration: float = 1.5,
    energy_threshold_db: float = -20,
    bark_padding: float = 0.05,
) -> List[np.ndarray]:
    """
    Extract features from isolated barks in an audio file.

    Args:
        file_path: Path to audio file
        min_bark_duration: Minimum bark duration in seconds
        max_bark_duration: Maximum bark duration in seconds
        energy_threshold_db: Energy threshold for bark detection (dB)
        bark_padding: Padding around detected barks (seconds)

    Returns:
        List of 40-dimensional feature vectors (one per detected bark)
    """
    try:
        sample_rate = 16000

        # Detect bark regions
        bark_regions = isolate_barks_energy(
            file_path,
            sample_rate=sample_rate,
            min_bark_duration=min_bark_duration,
            max_bark_duration=max_bark_duration,
            energy_threshold_db=energy_threshold_db,
        )

        if not bark_regions:
            # No barks detected, fall back to whole file
            features = extract_features(file_path)
            return [features] if features is not None else []

        # Load audio once for all segments
        y, sr = librosa.load(file_path, sr=sample_rate)

        # Extract features from each bark
        feature_list = []
        for start_time, end_time in bark_regions:
            segment = extract_bark_segment(y, start_time, end_time, sample_rate, bark_padding)
            if segment is not None:
                features = extract_features_from_audio(segment, sample_rate)
                if features is not None:
                    feature_list.append(features)

        # If no valid barks, fall back to whole file
        if not feature_list:
            features = extract_features(file_path)
            return [features] if features is not None else []

        return feature_list

    except Exception as e:
        print(f"Error processing {file_path} with bark isolation: {e}")
        return []

# --- Main Execution ---

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train a Random Forest classifier for dog breed classification")
parser.add_argument(
    "--use-dir-labels",
    action="store_true",
    help="Use directory names as class labels instead of extracting from filenames"
)
parser.add_argument(
    "--dataset-path",
    type=str,
    default="/Users/dillon_davis/repos/barkai/dogspeak_data2/dogspeak_released",
    help="Path to the dataset directory"
)
# Bark isolation options
parser.add_argument(
    "--enable-bark-isolation",
    action="store_true",
    help="Enable energy-based bark isolation to extract individual barks (0.3-1.5s) from clips"
)
parser.add_argument(
    "--bark-min-duration",
    type=float,
    default=0.3,
    help="Minimum bark duration in seconds (default: 0.3)"
)
parser.add_argument(
    "--bark-max-duration",
    type=float,
    default=1.5,
    help="Maximum bark duration in seconds (default: 1.5)"
)
parser.add_argument(
    "--bark-energy-threshold",
    type=float,
    default=-20,
    help="Energy threshold for bark detection in dB relative to max (default: -20)"
)
parser.add_argument(
    "--bark-padding",
    type=float,
    default=0.05,
    help="Padding around detected barks in seconds (default: 0.05)"
)
# Class balancing options
parser.add_argument(
    "--use-class-weights",
    action="store_true",
    help="Apply class weights to address class imbalance (weights inversely proportional to class frequency)"
)
args = parser.parse_args()

# 1. Setup Data Paths and Output Directories
dataset_path = args.dataset_path
checkpoint_base = Path("/Users/dillon_davis/repos/barkai/model_checkpoints")
features_dir = Path("/Users/dillon_davis/repos/barkai/features")

# Create directories
features_dir.mkdir(parents=True, exist_ok=True)

# Create timestamped output directory for model
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = checkpoint_base / f"random_forest_{timestamp}"
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Output directory: {output_dir}")

# Path for cached features (shared across runs, separate caches for each labeling strategy and bark isolation)
label_strategy = "dir_labels" if args.use_dir_labels else "filename_labels"
isolation_strategy = "bark_isolated" if args.enable_bark_isolation else "full_clip"
dataset_name = dataset_path.split('/')[-1].split('.')[0] or dataset_path.split('/')[-2].split('.')[0]
features_path = features_dir / f"mfcc_features_{label_strategy}_{isolation_strategy}_{dataset_name}.npz"

print(f"Label strategy: {'directory names' if args.use_dir_labels else 'filenames'}")
print(f"Bark isolation: {'enabled' if args.enable_bark_isolation else 'disabled'}")
if args.enable_bark_isolation:
    print(f"  - Min duration: {args.bark_min_duration}s")
    print(f"  - Max duration: {args.bark_max_duration}s")
    print(f"  - Energy threshold: {args.bark_energy_threshold} dB")
    print(f"  - Padding: {args.bark_padding}s")
print(f"Class weights: {'enabled (balanced)' if args.use_class_weights else 'disabled'}")

# Check if cached features exist
if features_path.exists():
    print(f"Loading cached features from {features_path}...")
    data = np.load(features_path)
    X = data['X']
    y = data['y']
    print("Features loaded from cache!")
else:
    print("Extracting features (this will be cached for future runs)...")
    X = []  # Features
    y = []  # Labels

    # Track statistics for bark isolation
    total_files = 0
    total_barks = 0
    files_with_no_barks = 0

    # Loop through each directory
    for dir_name in os.listdir(dataset_path):
        dir_path = os.path.join(dataset_path, dir_name)
        if not os.path.isdir(dir_path): continue

        for audio_file in tqdm(os.listdir(dir_path), desc=f"Processing {dir_name}"):
            if not audio_file.endswith('.wav'): continue

            # Determine class label based on labeling strategy
            if args.use_dir_labels:
                # Use directory name as class label
                label = dir_name.lower()
            else:
                # Extract breed from filename (default behavior)
                label = extract_breed_from_filename(audio_file)
                if label is None:
                    print(f"Warning: Could not extract breed from {audio_file}")
                    continue

            file_path = os.path.join(dir_path, audio_file)
            total_files += 1

            if args.enable_bark_isolation:
                # Extract features from isolated barks
                feature_list = extract_features_with_bark_isolation(
                    file_path,
                    min_bark_duration=args.bark_min_duration,
                    max_bark_duration=args.bark_max_duration,
                    energy_threshold_db=args.bark_energy_threshold,
                    bark_padding=args.bark_padding,
                )
                if feature_list:
                    total_barks += len(feature_list)
                    for features in feature_list:
                        X.append(features)
                        y.append(label)
                else:
                    files_with_no_barks += 1
            else:
                # Standard feature extraction (whole file)
                features = extract_features(file_path)
                if features is not None:
                    X.append(features)
                    y.append(label)

    X = np.array(X)
    y = np.array(y)

    # Print bark isolation statistics
    if args.enable_bark_isolation:
        print(f"\n--- Bark Isolation Statistics ---")
        print(f"Total files processed: {total_files}")
        print(f"Total barks extracted: {total_barks}")
        print(f"Average barks per file: {total_barks / max(total_files, 1):.2f}")
        print(f"Files with no barks detected: {files_with_no_barks}")

    # Save extracted features for future reuse
    print(f"\nSaving features to {features_path}...")
    np.savez(features_path, X=X, y=y)
    print("Features saved!")

print(f"\nDataset Shape: {X.shape}")  # Should be (Num_Samples, 40)
print(f"Unique classes found: {np.unique(y)}")
print(f"Samples per class: {dict(zip(*np.unique(y, return_counts=True)))}")

# Handle classes with only 1 sample by combining them into "other"
unique_classes, class_counts = np.unique(y, return_counts=True)
single_sample_classes = unique_classes[class_counts == 1]

if len(single_sample_classes) > 0:
    print("\n" + "=" * 80)
    print("⚠️  WARNING: CLASSES WITH ONLY 1 SAMPLE DETECTED ⚠️")
    print("=" * 80)
    print(f"The following {len(single_sample_classes)} classes have only 1 sample and will be")
    print("combined into an 'other' class to allow stratified train/test split:")
    print("-" * 80)
    for cls in sorted(single_sample_classes):
        print(f"  - {cls}")
    print("-" * 80)
    print(f"Total samples being relabeled as 'other': {len(single_sample_classes)}")
    print("=" * 80 + "\n")

    # Relabel single-sample classes as "other"
    y = np.array(["other" if label in single_sample_classes else label for label in y])

    # Print updated class distribution
    print("Updated class distribution after combining single-sample classes:")
    unique_updated, counts_updated = np.unique(y, return_counts=True)
    for cls, count in sorted(zip(unique_updated, counts_updated), key=lambda x: -x[1]):
        print(f"  {cls}: {count}")
    print()

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42, stratify=y)

# 3. Train Simple Baseline (Random Forest)
class_weight = "balanced" if args.use_class_weights else None
print(f"\nClass weights: {'balanced (inversely proportional to class frequency)' if args.use_class_weights else 'none'}")

clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weight)
clf.fit(X_train, y_train)

# 4. Save Model
model_path = output_dir / "random_forest_model.joblib"
print(f"\nSaving model to {model_path}...")
joblib.dump(clf, model_path)
print("Model saved!")

# 5. Evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\n--- Baseline Results ---")
print(f"Accuracy: {accuracy:.4f}")
print("\nDetailed Report:")
print(report)

# Save evaluation results
results_path = output_dir / "evaluation_results.txt"
with open(results_path, "w") as f:
    f.write(f"Random Forest Dog Breed Classification Results\n")
    f.write(f"Timestamp: {timestamp}\n")
    f.write(f"Label Strategy: {'directory names' if args.use_dir_labels else 'filenames'}\n")
    f.write(f"Bark Isolation: {'enabled' if args.enable_bark_isolation else 'disabled'}\n")
    if args.enable_bark_isolation:
        f.write(f"  - Min duration: {args.bark_min_duration}s\n")
        f.write(f"  - Max duration: {args.bark_max_duration}s\n")
        f.write(f"  - Energy threshold: {args.bark_energy_threshold} dB\n")
        f.write(f"  - Padding: {args.bark_padding}s\n")
    f.write(f"Class Weights: {'balanced' if args.use_class_weights else 'none'}\n")
    f.write(f"{'=' * 50}\n\n")
    f.write(f"Dataset Shape: {X.shape}\n")
    f.write(f"Unique classes: {list(np.unique(y))}\n\n")
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write(f"Detailed Classification Report:\n")
    f.write(report)

print(f"\nResults saved to {results_path}")
print(f"\nAll outputs saved to: {output_dir}")
