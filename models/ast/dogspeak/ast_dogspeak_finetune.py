#!/usr/bin/env python3
"""
BigAir Workflow: Fine-tune Audio Spectrogram Transformer (AST) on DogSpeak Dataset

This workflow fine-tunes the AST model pretrained on AudioSet for classifying dog breeds
based on their bark sounds using the DogSpeak dataset.

Features:
- Runs on A100XL GPU via BigQueue (single-GPU training)
- Downloads dataset tar from S3 and extracts it
- Parses breed labels from filenames (format: {integer}_{breed_name}_{gender}_dog_{integer}.wav)
- Class rebalancing via weighted sampling
- SpecAugment data augmentation
- Comprehensive evaluation and analysis

Usage:
    ba run relevance/frameworks/airlearner_v4/workflows/ast_dogspeak_finetune.py

Author: Dillon Davis
"""

from bigair import S3Path, step, workflow

# Conda environment with PyTorch, torchaudio, and transformers
CONDA_ENV = "production/ml_infra/ray/env:0.0.41"
CONDA_ENV = "aep/base:0.0.53"
CONDA_ENV = "production/relevance/pytorch/base_lightning:0.0.7"
CONDA_ENV = "devel/bark_classification:0.0.1"
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
    ],
    cacheable=False,
    auto_mlflow=True,
)
def train_ast_dogspeak(
    dataset_name: str,
    hf_token: str,
    model_name: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    output_s3_path: S3Path,
) -> S3Path:
    """
    Fine-tune AST model on DogSpeak dataset using a single A100XL GPU.

    Args:
        dataset_name: S3 path to the tar file containing the DogSpeak dataset
        hf_token: HuggingFace API token (unused, kept for compatibility)
        model_name: Pretrained AST model name
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Initial learning rate
        output_s3_path: S3 path to save model and artifacts

    Returns:
        S3Path to the saved model and results
    """
    import gc
    import json
    import os
    import random
    import warnings
    from collections import Counter
    from datetime import datetime
    from pathlib import Path
    from typing import Dict, List, Optional, Tuple

    import matplotlib
    import numpy as np
    import pandas as pd
    matplotlib.use('Agg')  # Non-interactive backend for headless
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

        # Model
        num_classes = 5

        # Audio processing
        sample_rate = 16000  # AST expects 16kHz
        max_duration = 10.0  # Maximum audio duration in seconds

        # Training - Optimized for A100XL
        weight_decay = 0.01
        warmup_ratio = 0.1
        gradient_accumulation_steps = 1
        max_grad_norm = 1.0
        use_mixed_precision = True
        # Enable parallel data loading to keep GPU fed
        # num_workers > 0 spawns subprocesses to prefetch batches while GPU computes
        # This prevents GPU starvation where it sits idle waiting for data
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

    # Scale learning rate with batch size (linear scaling rule)
    # Reference: "Accurate, Large Minibatch SGD" (Goyal et al., 2017)
    BASE_BATCH_SIZE = 64
    BASE_LR = 2.2e-4
    if learning_rate == BASE_LR:
        # Auto-scale LR if using default
        scaled_lr = BASE_LR * (batch_size / BASE_BATCH_SIZE)
        # Cap at reasonable maximum to avoid instability
        config.learning_rate = min(scaled_lr, 1e-2)
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
    local_output_dir = Path("/tmp/ast_dogspeak_output")
    local_output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Dataset Classes
    # =========================================================================

    BREED_TO_IDX = {
        "chihuahua": 0,
        "german_shepherd": 1,
        "husky": 2,
        "pitbull": 3,
        "shiba_inu": 4,
    }
    IDX_TO_BREED = {v: k for k, v in BREED_TO_IDX.items()}

    # Mapping for breed name variants (handles filenames without underscores)
    BREED_ALIASES = {
        "shibainu": "shiba_inu",
        "shiba_inu": "shiba_inu",
        "germanshepherd": "german_shepherd",
        "german_shepherd": "german_shepherd",
        "gsd": "german_shepherd",  # German Shepherd Dog abbreviation
        "chihuahua": "chihuahua",
        "husky": "husky",
        "pitbull": "pitbull",
    }

    class DogSpeakDataset(Dataset):
        """DogSpeak Dataset for dog breed classification from bark sounds"""

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
                audio_data, sr = sf.read(path, dtype='float32')
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
                waveform = waveform[:, start:start + max_samples]
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
    # Load Dataset from S3 (tar file)
    # =========================================================================

    print("\n" + "=" * 60)
    print("Loading DogSpeak Dataset from S3 (tar file)")
    print("=" * 60)

    import subprocess
    import tarfile

    data_path = Path("/tmp/dogspeak_data")
    data_path.mkdir(parents=True, exist_ok=True)
    tar_path = Path("/tmp/dogspeak_released.tar")

    print(f"Downloading dataset tar from {dataset_name}...")
    # Use aws s3 cp to download the tar file
    result = subprocess.run(
        ["aws", "s3", "cp", dataset_name, str(tar_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"S3 cp stderr: {result.stderr}")
        raise RuntimeError(f"Failed to download dataset from S3: {result.stderr}")
    print("Dataset tar downloaded successfully!")

    # Extract the tar file
    print(f"Extracting tar file to {data_path}...")
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=data_path)
    print("Tar extraction complete!")

    # Clean up tar file to save space
    tar_path.unlink()

    # =========================================================================
    # Prepare Data - Parse breed labels from filenames
    # =========================================================================

    print("\n" + "-" * 40)
    print("PREPARING DATA")
    print("-" * 40)

    def parse_breed_from_filename(filename: str) -> tuple:
        """
        Parse breed name and dog_id from filename.

        Filename format: {integer}_{breed_name}_{gender}_dog_{integer}.wav
        Examples:
            - 1_chihuahua_male_dog_1.wav -> breed=chihuahua, dog_id=1_chihuahua_male
            - 2_german_shepherd_female_dog_1.wav -> breed=german_shepherd, dog_id=2_german_shepherd_female

        Returns:
            tuple: (breed_name, dog_id) or (None, None) if parsing fails
        """
        try:
            # Remove .wav extension
            name = filename.replace('.wav', '')

            # Split by '_dog_' to separate breed/gender part from final integer
            parts = name.split('_dog_')
            if len(parts) != 2:
                return None, None

            prefix = parts[0]  # e.g., "1_chihuahua_male" or "2_german_shepherd_female"

            # Split prefix by '_': first is integer, last is gender, middle is breed
            prefix_parts = prefix.split('_')
            if len(prefix_parts) < 3:
                return None, None

            # Breed is everything between first integer and last gender
            breed = '_'.join(prefix_parts[1:-1])

            # Use prefix as dog_id (unique per dog)
            dog_id = prefix

            return breed, dog_id
        except Exception as e:
            print(f"Error parsing filename {filename}: {e}")
            return None, None

    # Find all WAV files and parse breed from filename
    # Filter out macOS metadata files (._*) first
    print("Scanning for WAV files and parsing breed labels from filenames...")
    all_wav_files = [f for f in data_path.rglob("*.wav") if not f.name.startswith("._")]
    print(f"Found {len(all_wav_files)} WAV files (excluding macOS metadata files)")

    audio_paths = []
    labels = []
    dog_ids = []
    skipped_files = []

    for wav_path in tqdm(all_wav_files, desc="Processing WAV files"):
        filename = wav_path.name
        breed, dog_id = parse_breed_from_filename(filename)

        if breed is None:
            skipped_files.append(filename)
            continue

        # Normalize breed name using alias mapping
        breed_lower = breed.lower().strip()
        breed_normalized = BREED_ALIASES.get(breed_lower, breed_lower)

        if breed_normalized not in BREED_TO_IDX:
            skipped_files.append(filename)
            continue

        label = BREED_TO_IDX[breed_normalized]
        audio_paths.append(str(wav_path))
        labels.append(label)
        dog_ids.append(dog_id)

    print(f"\nFound {len(audio_paths)} valid audio files")
    if skipped_files:
        print(f"Skipped {len(skipped_files)} files (unknown breed or invalid format)")
        if len(skipped_files) <= 10:
            print(f"  Skipped files: {skipped_files}")
        else:
            print(f"  First 10 skipped: {skipped_files[:10]}")

    # Show breed distribution
    label_counts = Counter(labels)
    print(f"\nBreed distribution:")
    for breed, idx in BREED_TO_IDX.items():
        count = label_counts.get(idx, 0)
        print(f"  {breed}: {count}")

    if len(audio_paths) == 0:
        raise ValueError("No audio files found! Check dataset structure.")

    split_df = pd.DataFrame({
        'path': audio_paths,
        'label': labels,
        'dog_id': dog_ids,
    })

    # Split by dog_id to prevent data leakage
    unique_dogs = split_df['dog_id'].unique()
    dog_breeds = split_df.groupby('dog_id')['label'].first()

    train_dogs, temp_dogs = train_test_split(
        unique_dogs,
        test_size=(config.val_ratio + config.test_ratio),
        stratify=[dog_breeds[d] for d in unique_dogs],
        random_state=config.seed,
    )

    val_dogs, test_dogs = train_test_split(
        temp_dogs,
        test_size=config.test_ratio / (config.val_ratio + config.test_ratio),
        stratify=[dog_breeds[d] for d in temp_dogs],
        random_state=config.seed,
    )

    train_mask = split_df['dog_id'].isin(train_dogs)
    val_mask = split_df['dog_id'].isin(val_dogs)
    test_mask = split_df['dog_id'].isin(test_dogs)

    train_data = [(row['path'], row['label']) for _, row in split_df[train_mask].iterrows()]
    val_data = [(row['path'], row['label']) for _, row in split_df[val_mask].iterrows()]
    test_data = [(row['path'], row['label']) for _, row in split_df[test_mask].iterrows()]

    # Calculate class weights
    train_labels = [l for _, l in train_data]
    class_counts = Counter(train_labels)
    total_samples = len(train_labels)

    class_weights = {}
    for cls_idx in range(config.num_classes):
        count = class_counts.get(cls_idx, 1)
        class_weights[cls_idx] = total_samples / (config.num_classes * count)

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_data)} samples ({len(train_dogs)} dogs)")
    print(f"  Val:   {len(val_data)} samples ({len(val_dogs)} dogs)")
    print(f"  Test:  {len(test_data)} samples ({len(test_dogs)} dogs)")

    # =========================================================================
    # Create DataLoaders
    # =========================================================================

    print("\nLoading AST feature extractor...")
    feature_extractor = ASTFeatureExtractor.from_pretrained(config.model_name)

    train_paths, train_labels_list = zip(*train_data)
    val_paths, val_labels_list = zip(*val_data)
    test_paths, test_labels_list = zip(*test_data)

    train_dataset = DogSpeakDataset(
        list(train_paths), list(train_labels_list),
        feature_extractor, config, augment=True
    )
    val_dataset = DogSpeakDataset(
        list(val_paths), list(val_labels_list),
        feature_extractor, config, augment=False
    )
    test_dataset = DogSpeakDataset(
        list(test_paths), list(test_labels_list),
        feature_extractor, config, augment=False
    )

    loader_kwargs = {
        "num_workers": config.num_workers,
        # Only use prefetch_factor and persistent_workers with multiple workers
        **({"prefetch_factor": config.prefetch_factor, "persistent_workers": True}
           if config.num_workers > 0 else {}),
        # pin_memory enables async CPU->GPU transfer while workers prepare next batch
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

    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, **loader_kwargs)

    # =========================================================================
    # Create Model
    # =========================================================================

    print("\n" + "-" * 40)
    print("LOADING MODEL")
    print("-" * 40)

    print(f"Loading AST model: {config.model_name}")
    model = ASTForAudioClassification.from_pretrained(
        config.model_name,
        num_labels=config.num_classes,
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

    # Clear any cached memory before loading model to GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    model = model.to(device)

    # Print GPU memory after model load
    if torch.cuda.is_available():
        print(f"GPU memory allocated after model load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # =========================================================================
    # Training Functions
    # =========================================================================

    def train_epoch(model, dataloader, optimizer, scheduler, class_weights_tensor, scaler):
        model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        # Move criterion to device once, not every batch
        criterion = nn.CrossEntropyLoss(
            weight=class_weights_tensor.to(device) if config.use_class_weights else None
        )

        progress_bar = tqdm(dataloader, desc="Training")
        optimizer.zero_grad(set_to_none=True)  # More memory efficient than zero_grad()

        for step_idx, batch in enumerate(progress_bar):
            # Use non_blocking=True with pin_memory for async CPU->GPU transfer
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

            # Store metrics before deleting tensors
            total_loss += loss.item() * config.gradient_accumulation_steps
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_labels.cpu().numpy())

            progress_bar.set_postfix({
                "loss": f"{loss.item() * config.gradient_accumulation_steps:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

            # Explicitly delete tensors to free memory
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

                # Free memory
                del input_values, batch_labels, outputs, logits, loss, probs, preds

        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted")
        precision_pc, recall_pc, f1_pc, support = precision_recall_fscore_support(all_labels, all_preds, average=None)

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predictions": all_preds,
            "labels": all_labels,
            "probabilities": np.array(all_probs),
            "precision_per_class": precision_pc,
            "recall_per_class": recall_pc,
            "f1_per_class": f1_pc,
            "support_per_class": support,
        }

    # =========================================================================
    # Training Loop
    # =========================================================================

    print("\n" + "-" * 40)
    print("TRAINING")
    print("-" * 40)

    class_weights_tensor = torch.tensor(
        [class_weights[i] for i in range(config.num_classes)],
        dtype=torch.float32,
    )

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

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

        # Clear memory at start of each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            print(f"GPU memory before epoch: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        train_results = train_epoch(model, train_loader, optimizer, scheduler, class_weights_tensor, scaler)

        # Clear memory between train and validation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        val_results = evaluate_model(model, val_loader, class_weights_tensor)

        history["train_loss"].append(train_results["loss"])
        history["train_accuracy"].append(train_results["accuracy"])
        history["val_loss"].append(val_results["loss"])
        history["val_accuracy"].append(val_results["accuracy"])

        # Log to MLflow
        mlflow.log_metrics({
            "train_loss": train_results["loss"],
            "train_accuracy": train_results["accuracy"],
            "val_loss": val_results["loss"],
            "val_accuracy": val_results["accuracy"],
            "val_f1": val_results["f1"],
        }, step=epoch)

        print(f"\nTrain Loss: {train_results['loss']:.4f} | Train Acc: {train_results['accuracy']:.4f}")
        print(f"Val Loss: {val_results['loss']:.4f} | Val Acc: {val_results['accuracy']:.4f} | Val F1: {val_results['f1']:.4f}")

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

    print(f"\nTest Accuracy: {test_results['accuracy']:.4f} ({test_results['accuracy']*100:.2f}%)")
    print(f"Test Precision: {test_results['precision']:.4f}")
    print(f"Test Recall: {test_results['recall']:.4f}")
    print(f"Test F1 Score: {test_results['f1']:.4f}")

    # Log final metrics to MLflow
    mlflow.log_metrics({
        "test_accuracy": test_results["accuracy"],
        "test_precision": test_results["precision"],
        "test_recall": test_results["recall"],
        "test_f1": test_results["f1"],
        "test_loss": test_results["loss"],
        "best_val_accuracy": best_val_accuracy,
    })

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

    # Confusion matrix
    cm = confusion_matrix(test_results["labels"], test_results["predictions"])
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    breed_names = [IDX_TO_BREED[i].replace("_", " ").title() for i in range(len(BREED_TO_IDX))]

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=breed_names, yticklabels=breed_names, ax=axes[0])
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title("Confusion Matrix (Counts)")

    sns.heatmap(cm_normalized, annot=True, fmt=".2%", cmap="Blues", xticklabels=breed_names, yticklabels=breed_names, ax=axes[1])
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_title("Confusion Matrix (Normalized)")

    plt.tight_layout()
    plt.savefig(local_output_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ROC curves
    n_classes = test_results["probabilities"].shape[1]
    labels_binary = np.zeros((len(test_results["labels"]), n_classes))
    for i, label in enumerate(test_results["labels"]):
        labels_binary[i, label] = 1

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(labels_binary[:, i], test_results["probabilities"][:, i])
        roc_auc = auc(fpr, tpr)
        breed_name = IDX_TO_BREED[i].replace("_", " ").title()
        ax.plot(fpr, tpr, color=colors[i], lw=2, label=f"{breed_name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=2, label="Random (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves per Class")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(local_output_dir / "roc_curves.png", dpi=300, bbox_inches="tight")
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
        "training_history": history,
        "config": {
            "model_name": config.model_name,
            "num_epochs": config.num_epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
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
def ast_dogspeak_finetune_workflow(
    dataset_name: str = "s3://airbnb-search-dev/user/dillon_davis/data/dogspeak_released.tar",
    hf_token: str = "",
    model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_epochs: int = 30,
    batch_size: int = 768,
    learning_rate: float = 2.2e-4,
    output_s3_bucket: str = "airbnb-search-dev",
    output_s3_prefix: str = "user/dillon_davis/ast_dogspeak",
):
    """
    BigAir workflow for fine-tuning AST on DogSpeak dataset.

    Args:
        dataset_name: S3 path to the tar file containing the DogSpeak dataset
        hf_token: HuggingFace API token (unused, kept for compatibility)
        model_name: Pretrained AST model name
        num_epochs: Number of training epochs
        batch_size: Training batch size (32 works well on A100XL)
        learning_rate: Initial learning rate
        output_s3_bucket: S3 bucket for output
        output_s3_prefix: S3 prefix for output
    """
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_s3_path = S3Path(bucket=output_s3_bucket, key=f"{output_s3_prefix}/{timestamp}")

    result_path = train_ast_dogspeak(
        dataset_name=dataset_name,
        hf_token=hf_token,
        model_name=model_name,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        output_s3_path=output_s3_path,
    )

    return result_path
