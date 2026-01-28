# Dog Bark Classification: A Multi-Dataset, Multi-Model Approach

## Table of Contents
1. [Overview](#overview)
2. [Application Architecture](#application-architecture)
3. [Datasets](#datasets)
4. [Modeling Methodologies](#modeling-methodologies)
5. [Results](#results)
6. [Discussion: Why AST Outperforms Random Forest](#discussion-why-ast-outperforms-random-forest)
7. [Conclusions](#conclusions)

---

## Overview

This project explores dog breed classification from audio bark recordings using two distinct datasets and two modeling approaches. The goal is to identify a dog's breed solely from the sound of its bark.

**Key Components:**
- **Streamlit Web Application** for real-time inference
- **Two Datasets**: DogSpeak (academic) and Custom YouTube (web-scraped)
- **Two Models**: Random Forest (MFCC features) and Fine-tuned Audio Spectrogram Transformer (AST)

---

## Application Architecture

### High-Level Design

```
+-------------------+     +--------------------+     +---------------------+
|   Audio Input     | --> |   Dog Detection    | --> |  Breed Classification|
|  (WAV/MP3/OGG)    |     | (MIT AST AudioSet) |     |   (RF or AST Model)  |
+-------------------+     +--------------------+     +---------------------+
                                   |                          |
                                   v                          v
                          +----------------+          +-----------------+
                          | Is Dog Bark?   |          | Breed Prediction|
                          | (Top-5 Check)  |          | + Confidence    |
                          +----------------+          +-----------------+
```

### Components

#### 1. Streamlit Frontend (`app.py`)
The application provides a web-based interface for users to:
- Upload audio files (WAV, MP3, OGG)
- Select from 4 breed classification models
- View dog detection confidence scores
- See breed prediction with probability distributions

#### 2. Dog Detection Stage
Uses MIT's Audio Spectrogram Transformer pre-trained on AudioSet (`MIT/ast-finetuned-audioset-10-10-0.4593`):
- Classifies 527 audio event categories
- Returns top-100 predictions with confidence scores
- Detects dog presence by checking if "Dog", "Bark", or "Puppy" appears in top-5 predictions

#### 3. Breed Classification Stage
Four model options available:

| Model | Dataset | Architecture | HuggingFace Repo |
|-------|---------|--------------|------------------|
| Random Forest (MFCC) | DogSpeak | sklearn RF | `dllndvs/dogspeak-breed-classifier` |
| AST Fine-tuned | DogSpeak | AST Transformer | `dllndvs/dogspeak-ast-breed-classifier` |
| YouTube RF (MFCC) | YouTube | sklearn RF | `dllndvs/dogspeak-youtube-breed-classifier` |
| YouTube AST | YouTube | AST Transformer | `dllndvs/dogspeak-youtube-ast-breed-classifier` |

### File Structure

```
barkai/
├── app.py                      # Main Streamlit application
├── train_random_forest.py      # RF model training script
├── finetune_ast_dogspeak.py    # AST fine-tuning script
├── inference_dogspeak.py       # Inference utilities
├── model_checkpoints/          # Trained model artifacts
├── features/                   # Cached MFCC feature extractions
├── dogspeak_data2/             # DogSpeak dataset
└── requirements.txt            # Python dependencies
```

---

## Datasets

### 1. DogSpeak Dataset

**Source:** Academic research dataset
**Format:** WAV audio files organized by individual dog ID
**Total Samples:** 77,202 audio clips

#### Dataset Statistics
- **Classes (Breeds):** 5 breeds
  - Chihuahua
  - German Shepherd (GSD)
  - Husky
  - Pitbull
  - Shiba Inu

- **Class Distribution:**
  | Breed | Samples |
  |-------|---------|
  | Husky | ~35,970 |
  | Shiba Inu | ~19,415 |
  | Chihuahua | ~8,155 |
  | GSD | ~7,550 |
  | Pitbull | ~6,112 |

#### File Naming Convention
```
{integer}_{breed_name}_{gender}_dog_{integer}.wav
Example: 58341_shibainu_F_dog_108.wav
```

#### Organization
- Files organized in directories by individual dog ID (dog_1, dog_2, ..., dog_156)
- 157 individual dogs total
- Breed labels extracted from filename

---

### 2. Custom YouTube Dataset

**Source:** Web-scraped from YouTube using yt-dlp
**Format:** MP3 audio files, preprocessed and filtered
**Total Raw Downloads:** 6,147 videos
**Total Processed Clips:** 84,169 audio segments
**Post-Filter (verified barks):** 19,319 samples

#### Collection Process

1. **YouTube Scraping** (`download_youtube_extended.py`)
   - Searches for 100 popular dog breeds
   - Uses 15 search query templates per breed:
     - "{breed} barking"
     - "{breed} bark"
     - "{breed} howling"
     - "{breed} growling"
     - "{breed} dog sounds"
     - ... and 10 more variations
   - Downloads up to 15 videos per search query
   - Filters: 3s < duration < 600s
   - Rate limiting to avoid detection

2. **Audio Preprocessing**
   - **Trim:** Simple silence removal (1,595 files)
   - **Extract Barks:** Isolate bark events using energy detection (1,337 files)
   - **Split:** Segment long recordings into 5-10s clips (3,215 files)
   - Total processing: 877,718s input → 526,723s output

3. **AST-Based Filtering**
   - Run MIT AST AudioSet classifier on all clips
   - Keep only clips where "Dog", "Bark", or "Puppy" in top-5 predictions
   - Pass rate: 19,282 of 84,169 clips (22.9%)

#### Dataset Statistics

- **Classes (Breeds):** 100 breeds
- **Total Audio Duration:** ~146 hours (post-filter)

**Sample Breed Distribution (filtered):**

| Breed | Samples |
|-------|---------|
| German Shepherd | 251 |
| Pekingese | 128 |
| Poodle | 128 |
| Rottweiler | 112 |
| Bulldog | 106 |
| Pug | 96 |
| Labrador Retriever | 78 |
| Siberian Husky | 75 |
| Chihuahua | 72 |
| Shih Tzu | 61 |
| ... | ... |

---

## Modeling Methodologies

### 1. Random Forest with MFCC Features

#### Feature Extraction
Mel-Frequency Cepstral Coefficients (MFCCs) capture the spectral characteristics of audio:

```python
def extract_features(file_path):
    # Load audio at 16kHz
    y, sr = librosa.load(file_path, sr=16000)

    # Remove silence
    y, _ = librosa.effects.trim(y)

    # Extract 20 MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    # Collapse time dimension
    mfcc_mean = np.mean(mfcc, axis=1)  # 20 values
    mfcc_std = np.std(mfcc, axis=1)     # 20 values

    # Final feature vector: 40 dimensions
    return np.concatenate([mfcc_mean, mfcc_std])
```

#### Feature Vector Structure
```
[MFCC_1_mean, MFCC_2_mean, ..., MFCC_20_mean, MFCC_1_std, MFCC_2_std, ..., MFCC_20_std]
```
- **Dimensions:** 40
- **Interpretation:**
  - Lower MFCCs: General spectral shape
  - Higher MFCCs: Finer spectral details
  - Mean: Average characteristics
  - Std: Temporal variation

#### Model Architecture
```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"  # Optional: addresses class imbalance
)
```

- **Ensemble Method:** 100 decision trees
- **Split Criterion:** Gini impurity
- **No max depth limit** (fully grown trees)
- **Class Weights (optional):** When enabled, weights are inversely proportional to class frequency

#### Training Pipeline
1. Extract MFCC features from all audio files
2. Cache features for reuse
3. Handle single-sample classes (combine into "other")
4. Train/test split: 80/20, stratified
5. Fit Random Forest classifier
6. Evaluate with classification report

---

### 2. Audio Spectrogram Transformer (AST)

#### Architecture Overview
The Audio Spectrogram Transformer adapts the Vision Transformer (ViT) architecture for audio:

```
Audio Waveform
      ↓
Mel Spectrogram (128 bins x time)
      ↓
Patch Embedding (16x16 patches)
      ↓
Position Embedding + CLS Token
      ↓
12x Transformer Encoder Layers
      ↓
CLS Token Output
      ↓
Classification Head
      ↓
Breed Predictions
```

#### Model Specifications
| Parameter | Value |
|-----------|-------|
| Base Model | MIT/ast-finetuned-audioset-10-10-0.4593 |
| Hidden Size | 768 |
| Attention Heads | 12 |
| Transformer Layers | 12 |
| Patch Size | 16x16 |
| Mel Bins | 128 |
| Max Length | 1024 frames |

#### Fine-tuning Strategy

1. **Transfer Learning**
   - Start from AudioSet pre-trained weights (527 classes)
   - Replace classification head for target classes
   - Fine-tune entire network

2. **Training Configuration**
   ```python
   optimizer = AdamW(lr=5e-5, weight_decay=0.01)
   scheduler = LinearWarmup + CosineDecay
   batch_size = 32 (base)
   epochs = 10
   ```

3. **Large Batch Training**
   - Linear learning rate scaling: `lr = base_lr * (batch_size / 32)`
   - Minimum warmup steps: max(100, calculated_warmup)

4. **Data Augmentation**
   - SpecAugment (optional): Time and frequency masking
   - Can be disabled for cleaner feature learning

#### Training Infrastructure
- **Platform:** BigAir + BigQueue (Airbnb ML platform)
- **Hardware:** NVIDIA A100 XL GPU (40GB)
- **Mixed Precision:** BF16 for numerical stability

---

## Results

### Model Comparison Summary

| Model | Dataset | Classes | Accuracy | Macro F1 | Improvement |
|-------|---------|---------|----------|----------|-------------|
| Random Forest (class weights) | DogSpeak | 5 | 73.71% | 0.63 | — |
| **AST Fine-tuned** | **DogSpeak** | **5** | **80.48%** | **0.81** | **+6.8%** |
| Random Forest | YouTube | 100 | 41.49% | 0.35 | — |
| **AST Fine-tuned** | **YouTube** | **100** | **45.77%** | **0.46** | **+4.3%** |

**Key Finding:** AST consistently outperforms Random Forest across both datasets, with larger gains on the cleaner DogSpeak dataset.

---

### Detailed Results

#### 1. Random Forest on DogSpeak (5 breeds, with class weights)

**Overall Performance:**
- **Accuracy:** 73.71%
- **Macro Precision:** 0.81
- **Macro Recall:** 0.57
- **Macro F1:** 0.63
- **Class Weights:** Balanced (inversely proportional to class frequency)

**Per-Class Performance:**

| Breed | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Chihuahua | 0.90 | 0.55 | 0.68 | 1,631 |
| GSD | 0.82 | 0.34 | 0.48 | 1,510 |
| Husky | 0.73 | 0.92 | 0.82 | 7,194 |
| Pitbull | 0.90 | 0.29 | 0.44 | 1,222 |
| Shiba Inu | 0.69 | 0.77 | 0.72 | 3,883 |

**Class Imbalance Correction:**

Class weights were applied to address the significant imbalance in the DogSpeak dataset (Husky has ~6x more samples than Pitbull). With `class_weight="balanced"`, sklearn computes weights as:
```
weight[class] = n_samples / (n_classes * n_samples_in_class)
```

**Effect of Class Weights:**
| Metric | Without Weights | With Weights | Change |
|--------|-----------------|--------------|--------|
| Accuracy | 74.18% | 73.71% | -0.47% |
| Macro Precision | 0.80 | 0.81 | +0.01 |
| Macro F1 | 0.62 | 0.63 | +0.01 |
| GSD Precision | 0.78 | 0.82 | +0.04 |
| Pitbull Recall | 0.26 | 0.29 | +0.03 |

**Key Observations:**
- Husky achieves best F1 (0.82) due to large sample size
- High precision across all classes (0.69-0.90)
- Class weights slightly improve minority class performance (GSD precision +0.04, Pitbull recall +0.03)
- Overall accuracy slightly decreases (-0.47%) as model trades majority class performance for minority class improvements
- Macro F1 improves slightly (+0.01), indicating more balanced performance across classes

---

#### 2. AST Fine-tuned on DogSpeak (5 breeds)

**Overall Performance:**
- **Accuracy:** 80.48%
- **Precision:** 81.54%
- **Recall:** 80.48%
- **F1-Score:** 80.81%
- **Test Loss:** 0.889

**Training Configuration:**
```json
{
  "architecture": "ASTForAudioClassification",
  "num_epochs": 30,
  "batch_size": 768,
  "learning_rate": 0.00022,
  "num_classes": 5
}
```

**Training Curves:**

The model shows consistent improvement over 30 epochs:
- Training loss decreases from 1.40 → 0.97
- Validation accuracy improves from 43% → 81%
- Best validation accuracy achieved at epoch 29: 80.83%

**Comparison with Random Forest (with class weights):**

| Metric | Random Forest | AST | Improvement |
|--------|---------------|-----|-------------|
| Accuracy | 73.71% | 80.48% | +6.77% |
| Macro F1 | 0.63 | 0.81 | +0.18 |
| Macro Precision | 0.81 | 0.82 | +0.01 |
| Macro Recall | 0.57 | 0.80 | +0.23 |

**Key Observations:**
- AST significantly improves recall across all classes
- More balanced performance across breeds compared to RF
- Training converges smoothly with cosine learning rate schedule

---

#### 3. Random Forest on YouTube (100 breeds)

**Overall Performance:**
- **Accuracy:** 41.49%
- **Macro Precision:** 0.38
- **Macro Recall:** 0.34
- **Macro F1:** 0.35

**Top Performing Breeds:**

| Breed | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| German Shepherd | 0.94 | 0.91 | **0.93** | 251 |
| Pekingese | 0.90 | 0.90 | **0.90** | 128 |
| Rottweiler | 0.70 | 0.84 | 0.76 | 112 |
| Shih Tzu | 0.74 | 0.70 | 0.72 | 61 |
| Basset Hound | 0.65 | 0.73 | 0.69 | 33 |
| Pomeranian | 0.68 | 0.62 | 0.65 | 45 |
| Pembroke Welsh Corgi | 0.69 | 0.61 | 0.65 | 54 |

**Challenging Breeds (Low Performance):**

| Breed | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Bulldog | 0.04 | 0.05 | 0.05 | 106 |
| Staffordshire Bull Terrier | 0.06 | 0.05 | 0.06 | 37 |
| Pug | 0.09 | 0.08 | 0.09 | 96 |
| Anatolian Shepherd | 0.10 | 0.08 | 0.09 | 13 |
| Australian Cattle Dog | 0.33 | 0.08 | 0.12 | 13 |

**Key Observations:**
- 100-class problem is significantly harder than 5-class
- Breeds with distinctive vocalizations perform well (German Shepherd, Pekingese)
- Brachycephalic breeds (Bulldog, Pug) are hard to classify - possibly similar bark characteristics
- Small sample sizes hurt performance (many breeds have <30 test samples)

---

#### 4. AST Fine-tuned on YouTube (100 breeds)

**Overall Performance:**
- **Accuracy:** 45.77%
- **Precision:** 51.57%
- **Recall:** 45.77%
- **F1-Score:** 46.26%
- **Test Loss:** 2.714

**Dataset Split:**
- Training: 13,497 samples
- Validation: 2,892 samples
- Test: 2,893 samples

**Training Configuration:**
```json
{
  "architecture": "ASTForAudioClassification",
  "num_epochs": 30,
  "batch_size": 768,
  "learning_rate": 0.000224,
  "num_classes": 100
}
```

**Training Curves:**

The model shows steady improvement over 30 epochs:
- Training loss decreases from 4.62 → 3.01
- Validation accuracy improves from 0.5% → 45.5%
- Best validation accuracy achieved at epoch 30: 45.54%

**Comparison with Random Forest:**

| Metric | Random Forest | AST | Improvement |
|--------|---------------|-----|-------------|
| Accuracy | 41.49% | 45.77% | +4.28% |
| Macro F1 | 0.35 | 0.46 | +0.11 |
| Macro Precision | 0.38 | 0.52 | +0.14 |
| Macro Recall | 0.34 | 0.46 | +0.12 |

**Key Observations:**
- AST achieves ~10% relative improvement over RF on this harder 100-class problem
- Still significant room for improvement (45.77% vs theoretical 100%)
- Noisy YouTube labels limit ceiling for both models
- Model continues to improve at epoch 30, suggesting longer training could help

---

## Discussion: Why AST Outperforms Random Forest

The Audio Spectrogram Transformer consistently outperforms the Random Forest baseline across both datasets. This section analyzes the architectural and methodological reasons for this performance gap.

### 1. Feature Representation Capacity

**Random Forest (MFCC):**
- Fixed 40-dimensional feature vector
- Hand-crafted features based on human auditory perception
- Collapses all temporal information into mean/std statistics
- Cannot capture fine-grained spectral patterns

**AST:**
- Learns hierarchical features from raw spectrograms
- 768-dimensional hidden representations per patch
- Millions of learnable parameters (87M total)
- Can discover task-specific discriminative features

```
Feature Dimensionality:
  MFCC:  40 dimensions (fixed)
  AST:   768 × num_patches (learned, adaptive)
```

### 2. Temporal Modeling

**Random Forest (MFCC):**
```
Audio → MFCC frames → mean(frames), std(frames) → 40-dim vector
```
- Completely discards temporal structure
- A bark at the start vs end produces identical features
- Cannot model bark duration, rhythm, or temporal patterns

**AST:**
```
Audio → Spectrogram → Patches → Positional Encoding → Transformer → Classification
```
- Preserves temporal information via positional embeddings
- Self-attention can model long-range dependencies
- Can learn that certain breeds have characteristic bark patterns (e.g., howl sequences)

### 3. Transfer Learning Advantage

**Random Forest:**
- Trains from scratch on target dataset only
- Limited by available labeled data
- No knowledge of general audio concepts

**AST:**
- Pre-trained on AudioSet: 2+ million audio clips, 527 classes
- Already understands concepts like "bark", "howl", "animal sound"
- Classification head fine-tunes on breed-specific patterns
- Effectively leverages orders of magnitude more training data

```
Training Data:
  RF:   77,202 samples (DogSpeak) or 19,319 samples (YouTube)
  AST:  2,000,000+ samples (AudioSet pre-training) + fine-tuning data
```

### 4. Attention Mechanism

The Transformer's self-attention allows AST to:
- Focus on discriminative bark segments within a recording
- Ignore background noise and irrelevant audio portions
- Learn which spectral regions distinguish different breeds

```
Example: German Shepherd classification
  - Attention may focus on low-frequency bark components
  - De-emphasize high-frequency background noise
  - Weight bark onset patterns more heavily
```

### 5. End-to-End Optimization

**Random Forest:**
- Features (MFCCs) are fixed, not optimized for the task
- Only decision boundaries are learned
- Suboptimal if MFCCs don't capture breed-discriminative information

**AST:**
- Entire pipeline is differentiable
- Feature extraction layers adapt to breed classification objective
- Learns optimal representations for the specific task

### 6. Quantitative Analysis

**Performance Gap by Dataset:**

| Dataset | RF Accuracy | AST Accuracy | Absolute Gain | Relative Gain |
|---------|-------------|--------------|---------------|---------------|
| DogSpeak (5 classes) | 73.71% | 80.48% | +6.77% | +9.2% |
| YouTube (100 classes) | 41.49% | 45.77% | +4.28% | +10.3% |

**Observations:**
- Relative improvement is larger on the harder 100-class problem
- AST's advantage grows as task complexity increases
- Cleaner data (DogSpeak) shows larger absolute improvement

### 7. When to Use Each Approach

**Use Random Forest when:**
- Interpretability is critical
- Compute resources are limited
- Quick prototyping is needed
- Small number of classes (<10)
- Training data is very limited (<1000 samples)

**Use AST when:**
- Maximum accuracy is the priority
- GPU resources are available
- Large number of classes (>10)
- Sufficient training data (>5000 samples)
- Transfer learning can be leveraged

---

## Conclusions

### Key Findings

1. **AST Consistently Outperforms Random Forest**
   - DogSpeak (5 breeds): AST achieves 80.48% vs RF 73.71% (+6.8% absolute)
   - YouTube (100 breeds): AST achieves 45.77% vs RF 41.49% (+4.3% absolute)
   - Relative improvement is larger on harder problems (+10.3% vs +9.2%)

2. **Dataset Quality Matters**
   - DogSpeak (curated academic dataset) yields better results than YouTube (web-scraped)
   - AST filtering helps but doesn't fully compensate for noisy labels
   - Clean labels enable AST to reach 80%+ accuracy on 5-class problem

3. **Class Count Impact**
   - 5-class problem: 80% accuracy achievable with AST
   - 100-class problem: Much harder, 46% accuracy represents reasonable performance
   - Both models struggle with fine-grained breed distinctions

4. **Transfer Learning is Critical**
   - AST's pre-training on AudioSet provides strong initialization
   - Random Forest cannot leverage external knowledge
   - The gap would likely widen with less training data

5. **Class Imbalance Affects Both Models**
   - Some breeds have distinctive vocalizations (German Shepherd: 93% F1 with RF)
   - Others are challenging regardless of samples (Bulldog: 5% F1)
   - AST shows more balanced performance across classes
   - Class weights in RF provide modest improvements for minority classes (+0.03 recall for Pitbull)
   - Trade-off: class weights slightly reduce overall accuracy (-0.47%) but improve macro F1 (+0.01)

### Final Model Comparison

| Aspect | Random Forest | AST |
|--------|---------------|-----|
| Best DogSpeak Accuracy | 73.71% (with class weights) | **80.48%** |
| Best YouTube Accuracy | 41.49% | **45.77%** |
| Training Time | Minutes | Hours |
| Inference Time | <1ms | ~100ms |
| Model Size | 200-1000 MB | 345 MB |
| GPU Required | No | Yes |
| Interpretability | High | Low |
| Class Imbalance Handling | Class weights (optional) | Weighted sampling |

### Recommendations for Improvement

1. **Data Collection**
   - Increase samples for underrepresented breeds
   - Use more sophisticated audio filtering (VAD, bark-specific detection)
   - Consider data augmentation (pitch shift, time stretch, mixup)

2. **Model Improvements**
   - Train AST for more epochs on YouTube (still improving at epoch 30)
   - Ensemble AST + RF predictions for robustness
   - Hierarchical classification (breed group → specific breed)
   - Try other audio transformers (HuBERT, Wav2Vec2, Data2Vec-Audio)

3. **Application Enhancements**
   - Add confidence thresholds for uncertain predictions
   - Provide "top-3" breed suggestions
   - Include breed group fallback for low-confidence predictions
   - Attention visualization for model interpretability

---

## Appendix

### A. HuggingFace Model Cards

| Model | Repo ID | Description |
|-------|---------|-------------|
| Dog Detector | `MIT/ast-finetuned-audioset-10-10-0.4593` | AudioSet pre-trained AST |
| DogSpeak RF | `dllndvs/dogspeak-breed-classifier` | 5-breed RF classifier |
| DogSpeak AST | `dllndvs/dogspeak-ast-breed-classifier` | 5-breed AST classifier |
| YouTube RF | `dllndvs/dogspeak-youtube-breed-classifier` | 100-breed RF classifier |
| YouTube AST | `dllndvs/dogspeak-youtube-ast-breed-classifier` | 100-breed AST classifier |

### B. Dependencies

```txt
transformers>=4.57.0
torch>=2.0.0
librosa>=0.10.0
scikit-learn>=1.3.0
streamlit>=1.28.0
huggingface_hub>=0.19.0
joblib>=1.3.0
numpy>=1.24.0
```

### C. YouTube Search Templates

```python
SEARCH_TEMPLATES = [
    "{breed} barking",
    "{breed} bark",
    "{breed} barking sound",
    "{breed} howling",
    "{breed} growling",
    "{breed} dog barking",
    "{breed} puppy barking",
    "{breed} angry barking",
    "{breed} loud bark",
    "{breed} dog sounds",
    "{breed} vocalization",
    "{breed} whining",
    "{breed} aggressive bark",
    "{breed} guard dog barking",
    "{breed} alert barking",
]
```

### D. 100 YouTube Dataset Breeds

Airedale Terrier, Akita, Alaskan Malamute, American Staffordshire Terrier, Anatolian Shepherd Dog, Australian Cattle Dog, Australian Shepherd, Basenji, Basset Hound, Beagle, Beauceron, Belgian Malinois, Bernese Mountain Dog, Bichon Frise, Biewer Terrier, Bloodhound, Border Collie, Border Terrier, Boston Terrier, Boxer, Boykin Spaniel, Brittany, Brussels Griffon, Bull Terrier, Bulldog, Bullmastiff, Cairn Terrier, Cane Corso, Cardigan Welsh Corgi, Cavalier King Charles Spaniel, Chesapeake Bay Retriever, Chihuahua, Chinese Crested, Chinese Shar-Pei, Chow Chow, Cocker Spaniel, Collie, Coton de Tulear, Dachshund, Dalmatian, Doberman Pinscher, Dogo Argentino, Dogue de Bordeaux, English Cocker Spaniel, English Setter, English Springer Spaniel, Flat-Coated Retriever, French Bulldog, German Shepherd Dog, German Shorthaired Pointer, German Wirehaired Pointer, Giant Schnauzer, Golden Retriever, Great Dane, Great Pyrenees, Great Swiss Mountain Dog, Havanese, Irish Setter, Irish Wolfhound, Italian Greyhound, Keeshond, Labrador Retriever, Lagotto Romagnolo, Leonberger, Lhasa Apso, Maltese, Mastiff, Miniature American Shepherd, Miniature Pinscher, Miniature Schnauzer, Newfoundland, Nova Scotia Duck Tolling Retriever, Old English Sheepdog, Papillon, Pekingese, Pembroke Welsh Corgi, Pomeranian, Poodle, Portuguese Water Dog, Pug, Rat Terrier, Rhodesian Ridgeback, Rottweiler, Russell Terrier, Saint Bernard, Samoyed, Scottish Terrier, Shetland Sheepdog, Shiba Inu, Shih Tzu, Siberian Husky, Soft Coated Wheaten Terrier, Staffordshire Bull Terrier, Standard Schnauzer, Vizsla, Weimaraner, West Highland White Terrier, Whippet, Wirehaired Pointing Griffon, Yorkshire Terrier
