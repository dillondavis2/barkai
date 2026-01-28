#!/usr/bin/env python3
"""
Filter Dog Bark Dataset using AudioSet-finetuned AST Model

This script uses the pretrained AST model (fine-tuned on AudioSet) to filter
the dataset to only keep clips that contain dog-related sounds.

Relevant AudioSet classes:
- Dog, Bark, Howl, Growling, Whimper (dog_whimper), Yip, Bow-wow
- Animal, Domestic animals/pets, Canidae (dogs, wolves)
- Squeak, Squeal (can be dog sounds)

Usage:
    python filter_with_ast.py --input_dir youtube_cleaned --output_dir youtube_filtered

Author: Dillon Davis
"""

import argparse
import json
import shutil
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")

# AudioSet class labels relevant to dog sounds
# These are the class names from AudioSet ontology
DOG_RELATED_CLASSES = {
    # Primary dog sounds
    "Dog",
    "Bark",
    "Howl",
    "Growling",
    "Whimper (dog)",
    "Yip",
    "Bow-wow",

    # Broader animal categories that include dogs
    "Animal",
    "Domestic animals, pets",
    "Canidae, dogs, wolves",

    # Other potentially relevant sounds
    "Squeak",
    "Squeal",
    "Whimper",
    "Crying, sobbing",  # Dogs can make crying sounds
    "Whine",
    "Snarl",

    # Howling variants
    "Bay",
    "Yowl",

    # Keep general mammal sounds as they might be dogs
    "Wild animals",

    # Specific vocalizations
    "Wail, moan",
}

# Additional class IDs to include (AudioSet specific)
# These are backup IDs in case name matching fails
DOG_RELATED_CLASS_IDS = {
    0,    # Speech (might have dog sounds in background)
    70,   # Animal
    71,   # Domestic animals, pets
    72,   # Dog
    73,   # Bark
    74,   # Howl
    75,   # Bow-wow
    76,   # Growling
    77,   # Whimper (dog)
    78,   # Cat (exclude)
    79,   # Purr (exclude)
    80,   # Meow (exclude)
    81,   # Hiss (exclude)
    82,   # Caterwaul (exclude)
    83,   # Livestock, farm animals, working animals
    84,   # Horse
    # ... more IDs can be added
}


def load_audioset_labels() -> Dict[int, str]:
    """
    Load AudioSet class labels.
    Returns a mapping from class index to class name.
    """
    # AudioSet class labels (527 classes)
    # This is a subset - the full list would be loaded from the model config
    # For now, we'll use the model's id2label mapping
    return {}


def get_dog_related_indices(id2label: Dict[int, str]) -> Set[int]:
    """
    Get indices of dog-related classes from the model's label mapping.
    """
    dog_indices = set()

    # Convert class names to lowercase for matching
    dog_classes_lower = {c.lower() for c in DOG_RELATED_CLASSES}

    for idx, label in id2label.items():
        label_lower = label.lower()

        # Check for exact or partial matches
        if any(dog_class in label_lower for dog_class in dog_classes_lower):
            dog_indices.add(idx)
            continue

        # Check for specific keywords
        keywords = ["dog", "bark", "howl", "growl", "whimper", "yip", "canid",
                   "bow-wow", "snarl", "yelp", "woof"]
        if any(kw in label_lower for kw in keywords):
            dog_indices.add(idx)

    return dog_indices


def load_model(model_name: str, device: str):
    """Load the AST model and feature extractor."""
    from transformers import ASTForAudioClassification, ASTFeatureExtractor

    print(f"Loading model: {model_name}")
    feature_extractor = ASTFeatureExtractor.from_pretrained(model_name)
    model = ASTForAudioClassification.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    return model, feature_extractor


def load_audio(file_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load and resample audio file."""
    try:
        import librosa
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
        return audio, sr
    except Exception as e:
        return None, str(e)


def predict_audio_class(
    audio: np.ndarray,
    sr: int,
    model,
    feature_extractor,
    device: str,
    top_k: int = 10,
) -> List[Tuple[int, str, float]]:
    """
    Predict audio class using AST model.

    Returns:
        List of (class_index, class_name, probability) tuples
    """
    # Prepare input
    inputs = feature_extractor(
        audio,
        sampling_rate=sr,
        return_tensors="pt",
    )
    input_values = inputs.input_values.to(device)

    # Run inference
    with torch.no_grad():
        outputs = model(input_values)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)

    # Get top-k predictions
    top_probs, top_indices = torch.topk(probs[0], k=top_k)

    results = []
    for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
        class_name = model.config.id2label.get(int(idx), f"Unknown_{idx}")
        results.append((int(idx), class_name, float(prob)))

    return results


def is_dog_sound(
    predictions: List[Tuple[int, str, float]],
    dog_indices: Set[int],
    threshold: float = 0.1,
    require_top_n: int = 5,
) -> Tuple[bool, str, float]:
    """
    Determine if the audio contains dog sounds based on predictions.

    Args:
        predictions: List of (class_index, class_name, probability)
        dog_indices: Set of class indices that are dog-related
        threshold: Minimum probability threshold for dog class
        require_top_n: Check if dog class is in top N predictions

    Returns:
        Tuple of (is_dog_sound, best_dog_class, probability)
    """
    # Check if any dog-related class is in top predictions with sufficient probability
    for i, (idx, class_name, prob) in enumerate(predictions[:require_top_n]):
        if idx in dog_indices and prob >= threshold:
            return True, class_name, prob

    # Also check if any dog class has probability above threshold anywhere
    for idx, class_name, prob in predictions:
        if idx in dog_indices and prob >= threshold:
            return True, class_name, prob

    return False, "", 0.0


def filter_dataset(
    input_dir: Path,
    output_dir: Path,
    model_name: str,
    threshold: float = 0.1,
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
    batch_size: int = 1,
    dry_run: bool = False,
) -> Dict:
    """
    Filter dataset to only keep clips with dog sounds.
    """
    print("\n" + "=" * 60)
    print("FILTERING DATASET WITH AST MODEL")
    print("=" * 60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Threshold: {threshold}")
    print(f"Dry run: {dry_run}")

    # Load model
    model, feature_extractor = load_model(model_name, device)

    # Get dog-related class indices
    id2label = model.config.id2label
    dog_indices = get_dog_related_indices(id2label)

    print(f"\nDog-related classes found ({len(dog_indices)}):")
    for idx in sorted(dog_indices):
        print(f"  {idx}: {id2label.get(idx, 'Unknown')}")

    # Collect all audio files
    all_files = []
    for breed_dir in sorted(input_dir.iterdir()):
        if breed_dir.is_dir():
            for wav_file in breed_dir.glob("*.wav"):
                all_files.append((breed_dir.name, wav_file))

    print(f"\nTotal files to process: {len(all_files)}")

    # Process files
    stats = {
        "total_input": len(all_files),
        "kept": 0,
        "filtered_out": 0,
        "errors": 0,
        "by_breed": defaultdict(lambda: {"input": 0, "kept": 0}),
        "by_class": defaultdict(int),
        "filtered_reasons": defaultdict(int),
    }

    kept_files = []
    filtered_files = []

    for breed, file_path in tqdm(all_files, desc="Classifying"):
        stats["by_breed"][breed]["input"] += 1

        # Load audio
        audio, sr = load_audio(str(file_path))
        if audio is None:
            stats["errors"] += 1
            continue

        # Get predictions
        try:
            predictions = predict_audio_class(
                audio, sr, model, feature_extractor, device
            )
        except Exception as e:
            stats["errors"] += 1
            continue

        # Check if it's a dog sound
        is_dog, dog_class, prob = is_dog_sound(predictions, dog_indices, threshold)

        if is_dog:
            stats["kept"] += 1
            stats["by_breed"][breed]["kept"] += 1
            stats["by_class"][dog_class] += 1
            kept_files.append({
                "breed": breed,
                "file": file_path.name,
                "path": str(file_path),
                "dog_class": dog_class,
                "probability": prob,
                "top_predictions": [(c, p) for _, c, p in predictions[:5]],
            })

            if not dry_run:
                # Copy file to output directory
                breed_out_dir = output_dir / breed
                breed_out_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, breed_out_dir / file_path.name)
        else:
            stats["filtered_out"] += 1
            top_class = predictions[0][1] if predictions else "Unknown"
            stats["filtered_reasons"][top_class] += 1
            filtered_files.append({
                "breed": breed,
                "file": file_path.name,
                "path": str(file_path),
                "top_class": top_class,
                "top_prob": predictions[0][2] if predictions else 0,
                "top_predictions": [(c, p) for _, c, p in predictions[:5]],
            })

    # Print summary
    print("\n" + "=" * 60)
    print("FILTERING SUMMARY")
    print("=" * 60)
    print(f"Total input files: {stats['total_input']}")
    print(f"Kept (dog sounds): {stats['kept']} ({stats['kept']/max(stats['total_input'],1)*100:.1f}%)")
    print(f"Filtered out: {stats['filtered_out']} ({stats['filtered_out']/max(stats['total_input'],1)*100:.1f}%)")
    print(f"Errors: {stats['errors']}")

    print("\nKept files by detected dog class:")
    for class_name, count in sorted(stats["by_class"].items(), key=lambda x: -x[1])[:15]:
        print(f"  {class_name}: {count}")

    print("\nTop reasons for filtering (non-dog classes):")
    for class_name, count in sorted(stats["filtered_reasons"].items(), key=lambda x: -x[1])[:15]:
        print(f"  {class_name}: {count}")

    print("\nBy breed (kept/input):")
    for breed in sorted(stats["by_breed"].keys()):
        b_stats = stats["by_breed"][breed]
        pct = b_stats["kept"] / max(b_stats["input"], 1) * 100
        print(f"  {breed}: {b_stats['kept']}/{b_stats['input']} ({pct:.1f}%)")

    # Save results (always save CSVs, even in dry_run mode)
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(kept_files).to_csv(output_dir / "kept_files.csv", index=False)
    pd.DataFrame(filtered_files).to_csv(output_dir / "filtered_files.csv", index=False)

    with open(output_dir / "filtering_stats.json", "w") as f:
        # Convert defaultdicts for JSON
        stats_json = {
            k: dict(v) if isinstance(v, defaultdict) else v
            for k, v in stats.items()
        }
        json.dump(stats_json, f, indent=2)

    print(f"\nResults saved to: {output_dir}")

    return stats


def copy_from_csv(csv_path: Path, output_dir: Path) -> None:
    """
    Copy files listed in kept_files.csv to output directory.
    This allows copying files after a dry_run without re-classifying.
    """
    import pandas as pd

    print("\n" + "=" * 60)
    print("COPYING FILES FROM CSV")
    print("=" * 60)
    print(f"CSV file: {csv_path}")
    print(f"Output directory: {output_dir}")

    df = pd.read_csv(csv_path)
    print(f"Total files to copy: {len(df)}")

    output_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    errors = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Copying"):
        src_path = Path(row["path"])
        breed = row["breed"]

        if not src_path.exists():
            print(f"  Warning: File not found: {src_path}")
            errors += 1
            continue

        breed_out_dir = output_dir / breed
        breed_out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, breed_out_dir / src_path.name)
        copied += 1

    print(f"\nCopied: {copied} files")
    print(f"Errors: {errors} files")
    print(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Filter dataset using AST model")
    parser.add_argument("--input_dir", type=str, default=None,
                       help="Input directory with audio files")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for filtered files")
    parser.add_argument("--model_name", type=str,
                       default="MIT/ast-finetuned-audioset-10-10-0.4593",
                       help="Pretrained AST model name")
    parser.add_argument("--threshold", type=float, default=0.1,
                       help="Minimum probability threshold for dog class")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda, mps, cpu)")
    parser.add_argument("--dry_run", action="store_true",
                       help="Preview filtering without copying files")
    parser.add_argument("--copy_only", type=str, default=None,
                       help="Path to kept_files.csv to copy files without re-classifying")

    args = parser.parse_args()

    # If copy_only mode, just copy files from CSV
    if args.copy_only:
        csv_path = Path(args.copy_only)
        if not csv_path.exists():
            print(f"Error: CSV file not found: {csv_path}")
            return

        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = csv_path.parent / "audio_files"

        copy_from_csv(csv_path, output_dir)
        return

    # Normal classification mode
    if not args.input_dir:
        print("Error: --input_dir is required unless using --copy_only")
        return

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir.parent / f"{input_dir.name}_filtered"

    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    filter_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        model_name=args.model_name,
        threshold=args.threshold,
        device=device,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
