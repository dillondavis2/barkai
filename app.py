import streamlit as st
from transformers import pipeline
import tempfile
import os
import numpy as np
import librosa
import joblib
from pathlib import Path
from huggingface_hub import hf_hub_download

# Hugging Face model repositories for breed classifiers
HF_RF_REPO_ID = "dllndvs/dogspeak-breed-classifier"
HF_RF_MODEL_FILENAME = "random_forest_model.joblib"
HF_AST_REPO_ID = "dllndvs/dogspeak-ast-breed-classifier"
HF_YT_RF_REPO_ID = "dllndvs/dogspeak-youtube-breed-classifier"
HF_YT_RF_MODEL_FILENAME = "random_forest_model.joblib"
HF_YT_AST_REPO_ID = "dllndvs/dogspeak-youtube-ast-breed-classifier"
LOCAL_MODEL_CACHE = Path("./model_cache")

# Model selection options
BREED_MODEL_OPTIONS = {
    "Random Forest (MFCC features)": "random_forest",
    "Dogspeak AST (Fine-tuned)": "ast",
    "YouTube Random Forest (MFCC features)": "youtube_random_forest",
    "YouTube AST (Fine-tuned)": "youtube_ast"
}

# --- Audio Processing ---

def save_audio_to_temp(audio_file):
    """Save uploaded audio file to a temporary file and return the path."""
    suffix = os.path.splitext(audio_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(audio_file.getvalue())
        return tmp_file.name


def extract_mfcc_features(file_path):
    """
    Extract MFCC features from audio file.
    Returns 40-dim vector: 20 MFCC means + 20 MFCC stds
    """
    try:
        y, sr = librosa.load(file_path, sr=16000)
        y, _ = librosa.effects.trim(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        return np.concatenate([mfcc_mean, mfcc_std]).reshape(1, -1)
    except Exception as e:
        st.warning(f"Error extracting features: {e}")
        return None


# --- Model Loaders (Cached) ---

@st.cache_resource
def load_dog_detector():
    """Load MIT's AST model fine-tuned on AudioSet for dog detection."""
    return pipeline("audio-classification", model="MIT/ast-finetuned-audioset-10-10-0.4593")


@st.cache_resource
def load_rf_breed_classifier():
    """Load Random Forest breed classifier from Hugging Face Hub."""
    try:
        model_path = hf_hub_download(
            repo_id=HF_RF_REPO_ID,
            filename=HF_RF_MODEL_FILENAME,
            cache_dir=LOCAL_MODEL_CACHE,
        )
        return joblib.load(model_path)
    except Exception as e:
        st.warning(f"Could not load Random Forest breed classifier: {e}")
        return None


@st.cache_resource
def load_ast_breed_classifier():
    """Load fine-tuned AST model for breed classification."""
    try:
        return pipeline("audio-classification", model=HF_AST_REPO_ID)
    except Exception as e:
        st.warning(f"Could not load AST breed classifier: {e}")
        return None


@st.cache_resource
def load_yt_rf_breed_classifier():
    """Load YouTube-trained Random Forest breed classifier from Hugging Face Hub."""
    try:
        model_path = hf_hub_download(
            repo_id=HF_YT_RF_REPO_ID,
            filename=HF_YT_RF_MODEL_FILENAME,
            cache_dir=LOCAL_MODEL_CACHE,
        )
        return joblib.load(model_path)
    except Exception as e:
        st.warning(f"Could not load YouTube Random Forest breed classifier: {e}")
        return None


@st.cache_resource
def load_yt_ast_breed_classifier():
    """Load YouTube fine-tuned AST model for breed classification."""
    try:
        return pipeline("audio-classification", model=HF_YT_AST_REPO_ID)
    except Exception as e:
        st.warning(f"Could not load YouTube AST breed classifier: {e}")
        return None


# --- Dog Detection ---

def detect_dog(audio_path, detector):
    """
    Run dog detection on audio file.
    Returns (is_detected, confidence, full_results)
    """
    results = detector(audio_path, top_k=100)
    top_labels = [result['label'] for result in results[:5]]
    scores = {result['label']: result['score'] for result in results[:5]}

    is_detected = any(
        keyword in label
        for label in top_labels
        for keyword in ['Dog', 'Bark', 'Puppy']
    )
    confidence = scores.get('Dog', scores.get('Bark', 0.0))

    return is_detected, confidence, results


def display_dog_detection_results(results):
    """Display detailed dog detection analysis as bar graph."""
    with st.expander("See detailed dog detection analysis"):
        for result in results:
            st.write(f"**{result['label']}:** {result['score']:.2%}")
            st.progress(result['score'])


# --- Breed Classification ---

def classify_breed_with_rf(audio_path, classifier):
    """
    Classify breed using Random Forest model with MFCC features.
    Returns (predicted_breed, confidence, sorted_results) or (None, None, None) on failure.
    """
    features = extract_mfcc_features(audio_path)
    if features is None:
        return None, None, None

    breed_pred = classifier.predict(features)[0]
    breed_proba = classifier.predict_proba(features)[0]
    breed_classes = classifier.classes_

    sorted_results = sorted(
        zip(breed_classes, breed_proba),
        key=lambda x: x[1],
        reverse=True
    )

    return breed_pred, max(breed_proba), sorted_results


def classify_breed_with_ast(audio_path, classifier):
    """
    Classify breed using AST model.
    Returns (predicted_breed, confidence, results_list) or (None, None, None) on failure.
    """
    results = classifier(audio_path, top_k=100)
    if not results:
        return None, None, None

    breed_pred = results[0]['label']
    confidence = results[0]['score']
    sorted_results = [(r['label'], r['score']) for r in results]

    return breed_pred, confidence, sorted_results


def display_breed_results(breed_pred, confidence, sorted_results):
    """Display breed classification results with progress bars."""
    breed_display = breed_pred.replace('_', ' ').title()
    st.info(f"🐕 **Predicted Breed: {breed_display}** (Confidence: {confidence:.2%})")

    with st.expander("See breed probabilities"):
        for breed, prob in sorted_results:
            breed_name = breed.replace('_', ' ').title()
            st.write(f"**{breed_name}:** {prob:.2%}")
            st.progress(prob)


def run_breed_classification(audio_path, model_type, model_name):
    """Run breed classification based on selected model type."""
    # Model type to loader and classifier function mapping
    model_config = {
        "random_forest": (load_rf_breed_classifier, classify_breed_with_rf),
        "ast": (load_ast_breed_classifier, classify_breed_with_ast),
        "youtube_random_forest": (load_yt_rf_breed_classifier, classify_breed_with_rf),
        "youtube_ast": (load_yt_ast_breed_classifier, classify_breed_with_ast),
    }

    loader, classify_fn = model_config[model_type]
    classifier = loader()

    if classifier is None:
        st.warning(f"{model_name} not available.")
        return

    breed_pred, confidence, sorted_results = classify_fn(audio_path, classifier)

    if breed_pred is None:
        st.warning("Could not classify breed.")
        return

    display_breed_results(breed_pred, confidence, sorted_results)


# --- Main App ---

def main():
    st.title("🐶 Dog Bark Detector & Breed Classifier")
    st.write("Upload an audio file to check if it contains a dog barking and identify the breed.")

    # Load dog detector
    dog_detector = load_dog_detector()

    # Model selection dropdown
    selected_model_name = st.selectbox(
        "Select Breed Classification Model",
        options=list(BREED_MODEL_OPTIONS.keys()),
        index=0
    )
    selected_model_type = BREED_MODEL_OPTIONS[selected_model_name]

    # File uploader
    audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "ogg"])

    if audio_file is None:
        return

    # Display audio player
    st.audio(audio_file)

    # Save to temp file
    tmp_path = save_audio_to_temp(audio_file)

    try:
        # Dog detection
        with st.spinner("Listening for barks..."):
            is_detected, dog_confidence, detection_results = detect_dog(tmp_path, dog_detector)

        if is_detected:
            st.success(f"✅ **Dog Detected!** (Confidence: {dog_confidence:.2%})")

            # Breed classification
            with st.spinner(f"Identifying breed using {selected_model_name}..."):
                run_breed_classification(tmp_path, selected_model_type, selected_model_name)
        else:
            st.error("🚫 **No Dog Detected.**")

        # Show detailed detection results
        display_dog_detection_results(detection_results)

    finally:
        # Clean up temp file
        os.remove(tmp_path)


if __name__ == "__main__":
    main()
