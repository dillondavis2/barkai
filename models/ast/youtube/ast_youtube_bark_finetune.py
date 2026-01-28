#!/usr/bin/env python3
"""
BigAir Workflow: Fine-tune Audio Spectrogram Transformer (AST) on YouTube Bark Dataset

This workflow fine-tunes the AST model pretrained on AudioSet for classifying dog breeds
based on their bark sounds using the YouTube bark dataset with AST-based filtering.

Features:
- Runs on A100XL GPU via BigQueue (single-GPU training)
- Downloads dataset tar from S3 and extracts it
- Pre-filters dataset using AST classification results (keeps only clips with Dog/Bark/Puppy in top-5)
- Optional bark isolation: extracts individual bark events (0.3-1.5s) from longer clips
- Uses breed subdirectory names as class labels (100 breeds)
- Class rebalancing via weighted sampling
- SpecAugment data augmentation
- Comprehensive evaluation and analysis

Usage:
    ba run relevance/frameworks/airlearner_v4/workflows/ast_youtube_bark_finetune.py

    # With bark isolation enabled:
    ba run relevance/frameworks/airlearner_v4/workflows/ast_youtube_bark_finetune.py -- --enable_bark_isolation True

Author: Dillon Davis
"""

from bigair import S3Path, step, workflow

# Conda environment with PyTorch, torchaudio, and transformers
CONDA_ENV = "production/guest_and_host/listing_intelligence/amenity_detector:0.0.12"


@step.bigqueue(
    num_worker=1,
    cpu=8,
    gpu=1,
    memory="64Gi",
    ephemeral_storage="200Gi",
    accelerator="nvidia-tesla-a100xl",
    conda_env=CONDA_ENV,
    pip_deps=[
        "huggingface_hub>=0.16.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "soundfile>=0.12.0",
        "librosa>=0.10.0",
    ],
    cacheable=False,
    auto_mlflow=True,
)
def train_ast_youtube_bark(
    dataset_tar_path: str,
    filter_json_path: str,
    model_name: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    enable_bark_isolation: bool,
    output_s3_path: S3Path,
) -> S3Path:
    """
    Fine-tune AST model on YouTube bark dataset with AST-based filtering.

    Args:
        dataset_tar_path: S3 path to the tar file containing the YouTube bark dataset
        filter_json_path: S3 path to the AST classification results JSON for filtering
        model_name: Pretrained AST model name
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Initial learning rate
        enable_bark_isolation: Whether to isolate individual barks from clips (0.3-1.5s)
        output_s3_path: S3 path to save model and artifacts

    Returns:
        S3Path to the saved model and results
    """
    import gc
    import json
    import os
    import random
    import subprocess
    import tarfile
    import warnings
    from collections import Counter
    from datetime import datetime
    from pathlib import Path
    from typing import Dict, List, Optional, Set, Tuple

    import matplotlib
    import numpy as np
    import pandas as pd

    matplotlib.use("Agg")  # Non-interactive backend for headless
    import matplotlib.pyplot as plt
    import mlflow
    import seaborn as sns
    import soundfile as sf
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchaudio
    import torchaudio.transforms as T
    from sklearn.metrics import (
        accuracy_score,
        auc,
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
        roc_curve,
    )
    from sklearn.model_selection import train_test_split
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    from tqdm import tqdm
    from transformers import (
        ASTFeatureExtractor,
        ASTForAudioClassification,
        get_linear_schedule_with_warmup,
    )

    warnings.filterwarnings("ignore")

    # =========================================================================
    # Configuration
    # =========================================================================

    class Config:
        """Training configuration"""

        # Audio processing
        sample_rate = 16000  # AST expects 16kHz
        max_duration = 10.0  # Maximum audio duration in seconds (reduced if bark isolation enabled)

        # Bark isolation parameters
        bark_isolation_enabled = False
        bark_min_duration = 0.3  # Minimum bark duration in seconds
        bark_max_duration = 1.5  # Maximum bark duration in seconds
        bark_energy_threshold_db = -20  # Energy threshold relative to max (dB)
        bark_padding = 0.05  # Padding around detected barks (seconds)
        bark_min_silence_between = 0.05  # Minimum silence between barks (seconds)

        # Training - Optimized for A100XL
        weight_decay = 0.01
        warmup_ratio = 0.1  # Base warmup ratio, will be scaled for large batches
        gradient_accumulation_steps = 1
        max_grad_norm = 1.0
        use_mixed_precision = True
        # Enable parallel data loading to keep GPU fed
        num_workers = 8  # Match cpu=8 in BigQueue config
        prefetch_factor = 4  # Prefetch 4 batches per worker
        # Enable gradient checkpointing to reduce activation memory
        gradient_checkpointing = True

        # Data splits
        train_ratio = 0.7
        val_ratio = 0.15
        test_ratio = 0.15

        # SpecAugment parameters
        freq_mask_param = 48
        time_mask_param = 192
        num_freq_masks = 2
        num_time_masks = 2

        # Class rebalancing
        use_class_weights = True
        use_weighted_sampler = True

        # Reproducibility
        seed = 42

    config = Config()
    # Set values from function parameters
    config.model_name = model_name
    config.batch_size = batch_size
    config.num_epochs = num_epochs
    config.bark_isolation_enabled = enable_bark_isolation

    # If bark isolation is enabled, reduce max duration since barks are short
    if enable_bark_isolation:
        config.max_duration = 2.0  # Isolated barks are 0.3-1.5s, allow some padding
        print(f"Bark isolation enabled - max_duration set to {config.max_duration}s")

    # Scale learning rate with batch size (linear scaling rule)
    # Reference: "Accurate, Large Minibatch SGD" (Goyal et al., 2017)
    BASE_BATCH_SIZE = 64
    BASE_LR = 2e-5
    if learning_rate == BASE_LR:
        # Auto-scale LR if using default
        scaled_lr = BASE_LR * (batch_size / BASE_BATCH_SIZE)
        # Cap at reasonable maximum to avoid instability
        config.learning_rate = min(scaled_lr, 1e-3)
        print(f"Auto-scaled learning rate: {BASE_LR} -> {config.learning_rate:.2e} (batch_size={batch_size})")
    else:
        # User specified custom LR, use as-is
        config.learning_rate = learning_rate

    # =========================================================================
    # Utility Functions
    # =========================================================================

    def set_seed(seed: int):
        """Set random seeds for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    set_seed(config.seed)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Create local output directory
    local_output_dir = Path("/tmp/ast_youtube_bark_output")
    local_output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Bark Isolation Functions
    # =========================================================================

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
        import librosa
        from scipy.ndimage import binary_dilation, binary_erosion

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
        audio_path: str,
        start_time: float,
        end_time: float,
        sample_rate: int = 16000,
        padding: float = 0.05,
    ) -> Optional[np.ndarray]:
        """
        Extract a single bark segment from an audio file.

        Args:
            audio_path: Path to audio file
            start_time: Start time of bark in seconds
            end_time: End time of bark in seconds
            sample_rate: Target sample rate
            padding: Padding around the bark in seconds

        Returns:
            Numpy array of audio samples, or None if extraction fails
        """
        import librosa

        try:
            y, sr = librosa.load(audio_path, sr=sample_rate)

            # Add padding around the bark
            start_sample = max(0, int((start_time - padding) * sr))
            end_sample = min(len(y), int((end_time + padding) * sr))

            segment = y[start_sample:end_sample]

            if len(segment) == 0:
                return None

            return segment

        except Exception as e:
            print(f"Error extracting bark segment from {audio_path}: {e}")
            return None

    def process_file_for_bark_isolation(
        audio_path: str,
        output_dir: Path,
        config: Config,
    ) -> List[str]:
        """
        Process a single audio file to extract isolated barks.

        Args:
            audio_path: Path to source audio file
            output_dir: Directory to save isolated barks
            config: Configuration object

        Returns:
            List of paths to saved bark files
        """
        bark_regions = isolate_barks_energy(
            audio_path,
            sample_rate=config.sample_rate,
            min_bark_duration=config.bark_min_duration,
            max_bark_duration=config.bark_max_duration,
            energy_threshold_db=config.bark_energy_threshold_db,
            min_silence_between=config.bark_min_silence_between,
        )

        if not bark_regions:
            return []

        saved_paths = []
        source_path = Path(audio_path)

        for idx, (start_time, end_time) in enumerate(bark_regions):
            segment = extract_bark_segment(
                audio_path,
                start_time,
                end_time,
                sample_rate=config.sample_rate,
                padding=config.bark_padding,
            )

            if segment is None or len(segment) == 0:
                continue

            # Save isolated bark
            bark_filename = f"{source_path.stem}_bark{idx:02d}.wav"
            bark_path = output_dir / bark_filename

            sf.write(bark_path, segment, config.sample_rate)
            saved_paths.append(str(bark_path))

        return saved_paths

    # =========================================================================
    # Download and Extract Dataset
    # =========================================================================

    print("\n" + "=" * 60)
    print("Downloading Dataset and Filter JSON")
    print("=" * 60)

    data_path = Path("/tmp/youtube_bark_data")
    data_path.mkdir(parents=True, exist_ok=True)
    tar_path = Path("/tmp/youtube_bark_audio.tar")
    filter_json_local = Path("/tmp/filter_results.json")

    # Download dataset tar
    print(f"Downloading dataset tar from {dataset_tar_path}...")
    result = subprocess.run(
        ["aws", "s3", "cp", dataset_tar_path, str(tar_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"S3 cp stderr: {result.stderr}")
        raise RuntimeError(f"Failed to download dataset from S3: {result.stderr}")
    print("Dataset tar downloaded successfully!")

    # Download filter JSON
    print(f"Downloading filter JSON from {filter_json_path}...")
    result = subprocess.run(
        ["aws", "s3", "cp", filter_json_path, str(filter_json_local)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"S3 cp stderr: {result.stderr}")
        raise RuntimeError(f"Failed to download filter JSON from S3: {result.stderr}")
    print("Filter JSON downloaded successfully!")

    # Extract the tar file
    print(f"Extracting tar file to {data_path}...")
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=data_path)
    print("Tar extraction complete!")

    # Clean up tar file to save space
    tar_path.unlink()

    # =========================================================================
    # Load Filter Results and Build Allowlist
    # =========================================================================

    print("\n" + "=" * 60)
    print("Loading Filter Results and Building Allowlist")
    print("=" * 60)

    with open(filter_json_local) as f:
        filter_results = json.load(f)

    print(f"Total entries in filter JSON: {len(filter_results)}")

    # Build set of file paths that pass the filter
    # Keep files where top-5 predictions contain "Dog", "Bark", or "Puppy"
    FILTER_KEYWORDS = ["Dog", "Bark", "Puppy"]

    def passes_filter(item: dict) -> bool:
        """Check if any top-5 prediction class contains Dog, Bark, or Puppy."""
        top_classes = [p["class"] for p in item.get("top_predictions", [])]
        return any(
            any(keyword in cls for keyword in FILTER_KEYWORDS) for cls in top_classes
        )

    # Build allowlist of file paths (relative paths from the JSON)
    # The JSON has paths like "youtube_cleaned/breed/filename.wav"
    # We need to match against extracted paths
    allowlist: Set[str] = set()
    for item in filter_results:
        if passes_filter(item):
            # Store just the filename for matching
            allowlist.add(item["filename"])

    print(f"Files passing filter (Dog/Bark/Puppy in top-5): {len(allowlist)}")

    # =========================================================================
    # Prepare Data - Scan for audio files and filter
    # =========================================================================

    print("\n" + "-" * 40)
    print("PREPARING DATA")
    print("-" * 40)

    # Find all WAV files (exclude macOS metadata files)
    print("Scanning for WAV files...")
    all_wav_files = [
        f for f in data_path.rglob("*.wav") if not f.name.startswith("._")
    ]
    print(f"Found {len(all_wav_files)} total WAV files")

    # Build breed to index mapping dynamically from directory structure
    # Expected structure: data_path/youtube_cleaned/BREED/filename.wav
    breed_set = set()
    for wav_path in all_wav_files:
        # Get breed from parent directory name
        breed = wav_path.parent.name
        if breed != "youtube_cleaned":  # Skip if we're at wrong level
            breed_set.add(breed)

    # Sort breeds for consistent indexing
    breeds_sorted = sorted(breed_set)
    BREED_TO_IDX = {breed: idx for idx, breed in enumerate(breeds_sorted)}
    IDX_TO_BREED = {idx: breed for breed, idx in BREED_TO_IDX.items()}
    num_classes = len(BREED_TO_IDX)

    print(f"Found {num_classes} breed classes")
    print(f"First 10 breeds: {breeds_sorted[:10]}")

    # Filter audio files and collect samples
    audio_paths = []
    labels = []
    skipped_not_in_allowlist = 0
    skipped_unknown_breed = 0

    for wav_path in tqdm(all_wav_files, desc="Filtering audio files"):
        filename = wav_path.name

        # Check if file passes the AST filter
        if filename not in allowlist:
            skipped_not_in_allowlist += 1
            continue

        # Get breed from parent directory
        breed = wav_path.parent.name
        if breed not in BREED_TO_IDX:
            skipped_unknown_breed += 1
            continue

        label = BREED_TO_IDX[breed]
        audio_paths.append(str(wav_path))
        labels.append(label)

    print(f"\nFiltering summary:")
    print(f"  Total WAV files: {len(all_wav_files)}")
    print(f"  Passed filter: {len(audio_paths)}")
    print(f"  Skipped (not in allowlist): {skipped_not_in_allowlist}")
    print(f"  Skipped (unknown breed): {skipped_unknown_breed}")

    if len(audio_paths) == 0:
        raise ValueError("No audio files passed the filter! Check dataset and filter JSON.")

    # =========================================================================
    # Optional: Bark Isolation
    # =========================================================================

    if config.bark_isolation_enabled:
        print("\n" + "=" * 60)
        print("BARK ISOLATION: Extracting Individual Barks")
        print("=" * 60)
        print(f"  Min bark duration: {config.bark_min_duration}s")
        print(f"  Max bark duration: {config.bark_max_duration}s")
        print(f"  Energy threshold: {config.bark_energy_threshold_db} dB")

        # Create temporary directory for isolated barks
        isolated_barks_dir = Path("/tmp/isolated_barks")
        isolated_barks_dir.mkdir(parents=True, exist_ok=True)

        # Process each file to extract isolated barks
        isolated_audio_paths = []
        isolated_labels = []
        files_with_barks = 0
        files_without_barks = 0

        for audio_path, label in tqdm(
            zip(audio_paths, labels), total=len(audio_paths), desc="Isolating barks"
        ):
            # Get breed for this file to organize isolated barks
            breed = IDX_TO_BREED[label]
            breed_output_dir = isolated_barks_dir / breed
            breed_output_dir.mkdir(parents=True, exist_ok=True)

            # Extract isolated barks from this file
            bark_paths = process_file_for_bark_isolation(
                audio_path, breed_output_dir, config
            )

            if bark_paths:
                files_with_barks += 1
                for bark_path in bark_paths:
                    isolated_audio_paths.append(bark_path)
                    isolated_labels.append(label)
            else:
                files_without_barks += 1

        print(f"\nBark isolation summary:")
        print(f"  Input files: {len(audio_paths)}")
        print(f"  Files with barks detected: {files_with_barks}")
        print(f"  Files without barks: {files_without_barks}")
        print(f"  Total isolated barks: {len(isolated_audio_paths)}")
        print(f"  Average barks per file: {len(isolated_audio_paths) / max(1, files_with_barks):.1f}")

        if len(isolated_audio_paths) == 0:
            raise ValueError(
                "No barks were isolated! Try adjusting bark isolation parameters "
                "(e.g., lower energy_threshold_db or adjust duration range)."
            )

        # Replace original paths with isolated bark paths
        audio_paths = isolated_audio_paths
        labels = isolated_labels

        # Log isolation stats
        mlflow.log_metrics({
            "bark_isolation_input_files": len(all_wav_files),
            "bark_isolation_output_barks": len(audio_paths),
            "bark_isolation_files_with_barks": files_with_barks,
            "bark_isolation_avg_barks_per_file": len(audio_paths) / max(1, files_with_barks),
        })

    # Show breed distribution after filtering (and optional isolation)
    label_counts = Counter(labels)
    print(f"\nBreed distribution (top 20):")
    for breed, idx in list(BREED_TO_IDX.items())[:20]:
        count = label_counts.get(idx, 0)
        print(f"  {breed}: {count}")

    # Log breed mapping for reference
    with open(local_output_dir / "breed_mapping.json", "w") as f:
        json.dump({"breed_to_idx": BREED_TO_IDX, "idx_to_breed": IDX_TO_BREED}, f, indent=2)

    # =========================================================================
    # Dataset Class
    # =========================================================================

    class YouTubeBarkDataset(Dataset):
        """YouTube Bark Dataset for dog breed classification"""

        def __init__(
            self,
            audio_paths: List[str],
            labels: List[int],
            feature_extractor: ASTFeatureExtractor,
            config: Config,
            augment: bool = False,
        ):
            self.audio_paths = audio_paths
            self.labels = labels
            self.feature_extractor = feature_extractor
            self.config = config
            self.augment = augment
            self.resampler_cache = {}

            if augment:
                self.freq_masking = T.FrequencyMasking(freq_mask_param=config.freq_mask_param)
                self.time_masking = T.TimeMasking(time_mask_param=config.time_mask_param)

        def __len__(self) -> int:
            return len(self.audio_paths)

        def _load_audio(self, path: str) -> Tuple[torch.Tensor, int]:
            """Load and preprocess audio file"""
            try:
                audio_data, sr = sf.read(path, dtype="float32")
                waveform = torch.from_numpy(audio_data)

                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0)
                else:
                    waveform = waveform.T
            except Exception as e:
                print(f"Error loading {path}: {e}")
                return torch.zeros(self.config.sample_rate * 2), self.config.sample_rate

            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            if sr != self.config.sample_rate:
                if sr not in self.resampler_cache:
                    self.resampler_cache[sr] = T.Resample(sr, self.config.sample_rate)
                waveform = self.resampler_cache[sr](waveform)

            max_samples = int(self.config.max_duration * self.config.sample_rate)
            if waveform.shape[1] > max_samples:
                if self.augment:
                    start = random.randint(0, waveform.shape[1] - max_samples)
                else:
                    start = (waveform.shape[1] - max_samples) // 2
                waveform = waveform[:, start : start + max_samples]
            elif waveform.shape[1] < max_samples:
                padding = max_samples - waveform.shape[1]
                waveform = F.pad(waveform, (0, padding))

            return waveform.squeeze(0), self.config.sample_rate

        def _apply_specaugment(self, mel_spec: torch.Tensor) -> torch.Tensor:
            """Apply SpecAugment to mel spectrogram"""
            spec = mel_spec.unsqueeze(0)
            for _ in range(self.config.num_freq_masks):
                spec = self.freq_masking(spec)
            for _ in range(self.config.num_time_masks):
                spec = self.time_masking(spec)
            return spec.squeeze(0)

        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            audio_path = self.audio_paths[idx]
            label = self.labels[idx]
            waveform, sr = self._load_audio(audio_path)

            inputs = self.feature_extractor(
                waveform.numpy(),
                sampling_rate=sr,
                return_tensors="pt",
            )
            input_values = inputs.input_values.squeeze(0)

            if self.augment:
                input_values = self._apply_specaugment(input_values)

            return {
                "input_values": input_values,
                "labels": torch.tensor(label, dtype=torch.long),
            }

    # =========================================================================
    # Split Data
    # =========================================================================

    # Create DataFrame for stratified splitting
    split_df = pd.DataFrame(
        {
            "path": audio_paths,
            "label": labels,
        }
    )

    # Stratified split
    train_df, temp_df = train_test_split(
        split_df,
        test_size=(config.val_ratio + config.test_ratio),
        stratify=split_df["label"],
        random_state=config.seed,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=config.test_ratio / (config.val_ratio + config.test_ratio),
        stratify=temp_df["label"],
        random_state=config.seed,
    )

    train_data = list(zip(train_df["path"].tolist(), train_df["label"].tolist()))
    val_data = list(zip(val_df["path"].tolist(), val_df["label"].tolist()))
    test_data = list(zip(test_df["path"].tolist(), test_df["label"].tolist()))

    # Calculate class weights
    train_labels = [l for _, l in train_data]
    class_counts = Counter(train_labels)
    total_samples = len(train_labels)

    class_weights = {}
    for cls_idx in range(num_classes):
        count = class_counts.get(cls_idx, 1)
        class_weights[cls_idx] = total_samples / (num_classes * count)

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Val:   {len(val_data)} samples")
    print(f"  Test:  {len(test_data)} samples")

    # =========================================================================
    # Create DataLoaders
    # =========================================================================

    print("\nLoading AST feature extractor...")
    feature_extractor = ASTFeatureExtractor.from_pretrained(config.model_name)

    train_paths, train_labels_list = zip(*train_data)
    val_paths, val_labels_list = zip(*val_data)
    test_paths, test_labels_list = zip(*test_data)

    train_dataset = YouTubeBarkDataset(
        list(train_paths),
        list(train_labels_list),
        feature_extractor,
        config,
        augment=True,
    )
    val_dataset = YouTubeBarkDataset(
        list(val_paths),
        list(val_labels_list),
        feature_extractor,
        config,
        augment=False,
    )
    test_dataset = YouTubeBarkDataset(
        list(test_paths),
        list(test_labels_list),
        feature_extractor,
        config,
        augment=False,
    )

    loader_kwargs = {
        "num_workers": config.num_workers,
        **(
            {"prefetch_factor": config.prefetch_factor, "persistent_workers": True}
            if config.num_workers > 0
            else {}
        ),
        "pin_memory": True if config.num_workers > 0 else False,
    }

    def create_weighted_sampler(labels: List[int], class_weights: Dict[int, float]):
        sample_weights = [class_weights[label] for label in labels]
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(labels),
            replacement=True,
        )

    if config.use_weighted_sampler:
        train_sampler = create_weighted_sampler(list(train_labels_list), class_weights)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            sampler=train_sampler,
            **loader_kwargs,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            **loader_kwargs,
        )

    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, **loader_kwargs
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, **loader_kwargs
    )

    # =========================================================================
    # Create Model
    # =========================================================================

    print("\n" + "-" * 40)
    print("LOADING MODEL")
    print("-" * 40)

    print(f"Loading AST model: {config.model_name}")
    model = ASTForAudioClassification.from_pretrained(
        config.model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    model.config.id2label = IDX_TO_BREED
    model.config.label2id = BREED_TO_IDX

    # Enable gradient checkpointing to reduce activation memory
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Number of classes: {num_classes}")

    # Clear any cached memory before loading model to GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    model = model.to(device)

    # Print GPU memory after model load
    if torch.cuda.is_available():
        print(
            f"GPU memory allocated after model load: {torch.cuda.memory_allocated() / 1e9:.2f} GB"
        )

    # =========================================================================
    # Training Functions
    # =========================================================================

    def train_epoch(model, dataloader, optimizer, scheduler, class_weights_tensor, scaler):
        model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        criterion = nn.CrossEntropyLoss(
            weight=class_weights_tensor.to(device) if config.use_class_weights else None
        )

        progress_bar = tqdm(dataloader, desc="Training")
        optimizer.zero_grad(set_to_none=True)

        for step_idx, batch in enumerate(progress_bar):
            input_values = batch["input_values"].to(device, non_blocking=True)
            batch_labels = batch["labels"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(input_values)
                logits = outputs.logits
                loss = criterion(logits, batch_labels)
                loss = loss / config.gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (step_idx + 1) % config.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * config.gradient_accumulation_steps
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_labels.cpu().numpy())

            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item() * config.gradient_accumulation_steps:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                }
            )

            del input_values, batch_labels, outputs, logits, loss

        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        return {"loss": avg_loss, "accuracy": accuracy}

    def evaluate_model(model, dataloader, class_weights_tensor):
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        criterion = nn.CrossEntropyLoss(
            weight=class_weights_tensor.to(device) if config.use_class_weights else None
        )

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_values = batch["input_values"].to(device, non_blocking=True)
                batch_labels = batch["labels"].to(device, non_blocking=True)

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(input_values)
                    logits = outputs.logits
                    loss = criterion(logits, batch_labels)

                total_loss += loss.item()
                probs = F.softmax(logits.float(), dim=-1)
                preds = torch.argmax(logits, dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                del input_values, batch_labels, outputs, logits, loss, probs, preds

        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="weighted", zero_division=0
        )

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predictions": all_preds,
            "labels": all_labels,
            "probabilities": np.array(all_probs),
        }

    # =========================================================================
    # Training Loop
    # =========================================================================

    print("\n" + "-" * 40)
    print("TRAINING")
    print("-" * 40)

    class_weights_tensor = torch.tensor(
        [class_weights.get(i, 1.0) for i in range(num_classes)],
        dtype=torch.float32,
    )

    optimizer = AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    total_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation_steps

    # Scale warmup for large batch training
    # With larger batches, we have fewer steps, so ensure adequate warmup
    # Minimum 100 steps or warmup_ratio of total, whichever is larger
    warmup_steps = max(100, int(total_steps * config.warmup_ratio))
    print(f"Training steps: {total_steps}, Warmup steps: {warmup_steps}")

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = torch.amp.GradScaler("cuda")

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    best_val_accuracy = 0.0
    best_model_state = None

    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        print("-" * 30)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            print(f"GPU memory before epoch: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        train_results = train_epoch(
            model, train_loader, optimizer, scheduler, class_weights_tensor, scaler
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        val_results = evaluate_model(model, val_loader, class_weights_tensor)

        history["train_loss"].append(train_results["loss"])
        history["train_accuracy"].append(train_results["accuracy"])
        history["val_loss"].append(val_results["loss"])
        history["val_accuracy"].append(val_results["accuracy"])

        mlflow.log_metrics(
            {
                "train_loss": train_results["loss"],
                "train_accuracy": train_results["accuracy"],
                "val_loss": val_results["loss"],
                "val_accuracy": val_results["accuracy"],
                "val_f1": val_results["f1"],
            },
            step=epoch,
        )

        print(
            f"\nTrain Loss: {train_results['loss']:.4f} | Train Acc: {train_results['accuracy']:.4f}"
        )
        print(
            f"Val Loss: {val_results['loss']:.4f} | Val Acc: {val_results['accuracy']:.4f} | Val F1: {val_results['f1']:.4f}"
        )

        if val_results["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_results["accuracy"]
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"New best model! Validation accuracy: {best_val_accuracy:.4f}")

            model.save_pretrained(local_output_dir / "best_model")
            feature_extractor.save_pretrained(local_output_dir / "best_model")

    # Load best model for final evaluation
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    # =========================================================================
    # Final Evaluation
    # =========================================================================

    print("\n" + "-" * 40)
    print("FINAL EVALUATION ON TEST SET")
    print("-" * 40)

    test_results = evaluate_model(model, test_loader, class_weights_tensor)

    print(
        f"\nTest Accuracy: {test_results['accuracy']:.4f} ({test_results['accuracy']*100:.2f}%)"
    )
    print(f"Test Precision: {test_results['precision']:.4f}")
    print(f"Test Recall: {test_results['recall']:.4f}")
    print(f"Test F1 Score: {test_results['f1']:.4f}")

    mlflow.log_metrics(
        {
            "test_accuracy": test_results["accuracy"],
            "test_precision": test_results["precision"],
            "test_recall": test_results["recall"],
            "test_f1": test_results["f1"],
            "test_loss": test_results["loss"],
            "best_val_accuracy": best_val_accuracy,
        }
    )

    # =========================================================================
    # Generate Visualizations
    # =========================================================================

    print("\n" + "-" * 40)
    print("GENERATING VISUALIZATIONS")
    print("-" * 40)

    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history["train_loss"], label="Train", marker="o", markersize=4)
    axes[0].plot(history["val_loss"], label="Validation", marker="s", markersize=4)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["train_accuracy"], label="Train", marker="o", markersize=4)
    axes[1].plot(history["val_accuracy"], label="Validation", marker="s", markersize=4)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(local_output_dir / "training_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Confusion matrix (top 20 classes by support for readability)
    test_label_counts = Counter(test_results["labels"])
    top_classes = [cls for cls, _ in test_label_counts.most_common(20)]

    # Filter to top classes for visualization
    mask = [l in top_classes for l in test_results["labels"]]
    filtered_labels = [l for l, m in zip(test_results["labels"], mask) if m]
    filtered_preds = [p for p, m in zip(test_results["predictions"], mask) if m]

    if filtered_labels:
        cm = confusion_matrix(filtered_labels, filtered_preds, labels=top_classes)
        cm_normalized = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)

        fig, ax = plt.subplots(figsize=(16, 14))
        breed_names = [IDX_TO_BREED[i].replace("_", " ")[:15] for i in top_classes]

        sns.heatmap(
            cm_normalized,
            annot=False,
            cmap="Blues",
            xticklabels=breed_names,
            yticklabels=breed_names,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix (Top 20 Classes, Normalized)")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        plt.tight_layout()
        plt.savefig(local_output_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
        plt.close()

    # Log artifacts to MLflow
    mlflow.log_artifacts(str(local_output_dir))

    # =========================================================================
    # Save Results
    # =========================================================================

    final_results = {
        "test_accuracy": float(test_results["accuracy"]),
        "test_precision": float(test_results["precision"]),
        "test_recall": float(test_results["recall"]),
        "test_f1": float(test_results["f1"]),
        "test_loss": float(test_results["loss"]),
        "best_val_accuracy": float(best_val_accuracy),
        "num_classes": num_classes,
        "num_train_samples": len(train_data),
        "num_val_samples": len(val_data),
        "num_test_samples": len(test_data),
        "training_history": history,
        "config": {
            "model_name": config.model_name,
            "num_epochs": config.num_epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "bark_isolation_enabled": config.bark_isolation_enabled,
            "bark_min_duration": config.bark_min_duration if config.bark_isolation_enabled else None,
            "bark_max_duration": config.bark_max_duration if config.bark_isolation_enabled else None,
            "bark_energy_threshold_db": config.bark_energy_threshold_db if config.bark_isolation_enabled else None,
        },
    }

    with open(local_output_dir / "results.json", "w") as f:
        json.dump(final_results, f, indent=2)

    # Copy to S3 using aws s3 sync
    s3_uri = f"s3://{output_s3_path.bucket}/{output_s3_path.key}"
    print(f"\nUploading results to {s3_uri}...")
    result = subprocess.run(
        ["aws", "s3", "sync", str(local_output_dir), s3_uri],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"S3 sync stderr: {result.stderr}")
        raise RuntimeError(f"Failed to upload results to S3: {result.stderr}")

    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE!")
    print(f"{'=' * 60}")
    print(f"Results saved to: {s3_uri}")

    return output_s3_path


@workflow
def ast_youtube_bark_finetune_workflow(
    dataset_tar_path: str = "s3://airbnb-search-dev/user/dillon_davis/data/youtube_bark_audio.tar",
    filter_json_path: str = "s3://airbnb-search-dev/user/dillon_davis/ast_bark_filter/20260127_002414/all_results.json",
    model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 2e-5,
    enable_bark_isolation: bool = False,
    output_s3_bucket: str = "airbnb-search-dev",
    output_s3_prefix: str = "user/dillon_davis/ast_youtube_bark",
):
    """
    BigAir workflow for fine-tuning AST on YouTube bark dataset with filtering.

    Args:
        dataset_tar_path: S3 path to the tar file containing the YouTube bark dataset
        filter_json_path: S3 path to AST classification results JSON for filtering
        model_name: Pretrained AST model name
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Initial learning rate
        enable_bark_isolation: Whether to isolate individual barks (0.3-1.5s) from clips
        output_s3_bucket: S3 bucket for output
        output_s3_prefix: S3 prefix for output
    """
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_s3_path = S3Path(bucket=output_s3_bucket, key=f"{output_s3_prefix}/{timestamp}")

    result_path = train_ast_youtube_bark(
        dataset_tar_path=dataset_tar_path,
        filter_json_path=filter_json_path,
        model_name=model_name,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        enable_bark_isolation=enable_bark_isolation,
        output_s3_path=output_s3_path,
    )

    return result_path
