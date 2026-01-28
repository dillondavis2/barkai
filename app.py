import streamlit as st
from transformers import pipeline
import tempfile
import os
import gc
import torch
import numpy as np
import librosa
import joblib
import logging
from pathlib import Path
from huggingface_hub import hf_hub_download

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Hugging Face model repositories
HF_DOG_DETECTOR_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
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


def clear_memory():
    """Aggressively clear memory after unloading models."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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


# --- Model Manager ---
# Only one breed classifier is kept in memory at a time to stay within memory limits

class BreedModelManager:
    """
    Manages breed classifier loading/unloading to prevent memory exhaustion.
    Only one breed classifier is kept in memory at a time.
    """

    def __init__(self):
        self._current_model = None
        self._current_model_type = None

    def get_model(self, model_type):
        """
        Get the breed classifier for the specified model type.
        Unloads any previously loaded model first to free memory.
        """
        # If the requested model is already loaded, return it
        if self._current_model_type == model_type and self._current_model is not None:
            logger.info(f"Reusing cached model: {model_type}")
            return self._current_model

        # Unload current model before loading new one
        self._unload_current_model()

        # Load the requested model
        logger.info(f"Loading breed classifier: {model_type}")
        model = self._load_model(model_type)

        if model is not None:
            self._current_model = model
            self._current_model_type = model_type

        return model

    def _unload_current_model(self):
        """Unload the currently loaded model and free memory."""
        if self._current_model is not None:
            logger.info(f"Unloading model: {self._current_model_type}")
            # Delete the model
            del self._current_model
            self._current_model = None
            self._current_model_type = None
            # Aggressively clear memory
            clear_memory()

    def _load_model(self, model_type):
        """Load a specific model type."""
        try:
            if model_type == "random_forest":
                return self._load_rf_model(HF_RF_REPO_ID, HF_RF_MODEL_FILENAME)
            elif model_type == "youtube_random_forest":
                return self._load_rf_model(HF_YT_RF_REPO_ID, HF_YT_RF_MODEL_FILENAME)
            elif model_type == "ast":
                return self._load_ast_model(HF_AST_REPO_ID)
            elif model_type == "youtube_ast":
                return self._load_ast_model(HF_YT_AST_REPO_ID)
            else:
                logger.error(f"Unknown model type: {model_type}")
                return None
        except Exception as e:
            logger.error(f"Error loading model {model_type}: {e}")
            return None

    def _load_rf_model(self, repo_id, filename):
        """Load a Random Forest model from HuggingFace Hub."""
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=LOCAL_MODEL_CACHE,
        )
        return joblib.load(model_path)

    def _load_ast_model(self, repo_id):
        """Load an AST model from HuggingFace Hub."""
        return pipeline("audio-classification", model=repo_id)


def get_breed_model_manager():
    """Get or create the breed model manager in session state."""
    if 'breed_model_manager' not in st.session_state:
        st.session_state.breed_model_manager = BreedModelManager()
    return st.session_state.breed_model_manager


# --- Dog Detector (always cached - required for all classifications) ---

@st.cache_resource
def load_dog_detector():
    """
    Load MIT's AST model fine-tuned on AudioSet for dog detection.
    This is cached because it's always needed regardless of breed classifier choice.
    """
    logger.info("Loading dog detector model...")
    model = pipeline("audio-classification", model=HF_DOG_DETECTOR_ID)
    logger.info("Dog detector model loaded successfully")
    return model


# --- Dog Detection ---

def detect_dog(audio_path, detector):
    """
    Run dog detection on audio file using local model.
    Returns (is_detected, confidence, full_results)
    """
    logger.info(f"Starting dog detection for: {audio_path}")

    try:
        results = detector(audio_path, top_k=100)
        logger.info(f"Dog detection returned {len(results)} results")

        top_labels = [result['label'] for result in results[:5]]
        scores = {result['label']: result['score'] for result in results[:5]}
        logger.info(f"Top labels: {top_labels}")

        is_detected = any(
            keyword in label
            for label in top_labels
            for keyword in ['Dog', 'Bark', 'Puppy']
        )
        confidence = scores.get('Dog', scores.get('Bark', 0.0))
        logger.info(f"Detection result: is_detected={is_detected}, confidence={confidence}")

        return is_detected, confidence, results

    except Exception as e:
        logger.error(f"Dog detection error: {type(e).__name__}: {e}")
        st.error(f"Dog detection error: {e}")
        return False, 0.0, []


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
    try:
        results = classifier(audio_path, top_k=100)

        if not results:
            return None, None, None

        breed_pred = results[0]['label']
        confidence = results[0]['score']
        sorted_results = [(r['label'], r['score']) for r in results]

        return breed_pred, confidence, sorted_results

    except Exception as e:
        logger.error(f"AST breed classification error: {type(e).__name__}: {e}")
        st.error(f"Breed classification error: {e}")
        return None, None, None


def display_breed_results(breed_pred, confidence, sorted_results):
    """Display breed classification results with progress bars."""
    breed_display = breed_pred.replace('_', ' ').title()
    st.info(f"**Predicted Breed: {breed_display}** (Confidence: {confidence:.2%})")

    with st.expander("See breed probabilities"):
        for breed, prob in sorted_results:
            breed_name = breed.replace('_', ' ').title()
            st.write(f"**{breed_name}:** {prob:.2%}")
            st.progress(prob)


def run_breed_classification(audio_path, model_type, model_name):
    """Run breed classification based on selected model type."""
    # Get the model manager (handles loading/unloading)
    model_manager = get_breed_model_manager()

    # Get the appropriate classifier (this will unload any previous model)
    classifier = model_manager.get_model(model_type)

    if classifier is None:
        st.warning(f"{model_name} not available.")
        return

    # Classify based on model type
    if model_type in ["random_forest", "youtube_random_forest"]:
        breed_pred, confidence, sorted_results = classify_breed_with_rf(audio_path, classifier)
    else:  # ast or youtube_ast
        breed_pred, confidence, sorted_results = classify_breed_with_ast(audio_path, classifier)

    if breed_pred is None:
        st.warning("Could not classify breed.")
        return

    display_breed_results(breed_pred, confidence, sorted_results)


# --- Main App ---

def main():
    st.title("Dog Bark Detector & Breed Classifier")
    st.write("Upload an audio file to check if it contains a dog barking and identify the breed.")

    # Show memory optimization note
    st.caption("Memory optimized: Only one breed classifier is loaded at a time.")

    # Load dog detector (always needed, cached)
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
            st.success(f"**Dog Detected!** (Confidence: {dog_confidence:.2%})")

            # Breed classification
            with st.spinner(f"Identifying breed using {selected_model_name}..."):
                run_breed_classification(tmp_path, selected_model_type, selected_model_name)
        else:
            st.error("**No Dog Detected.**")

        # Show detailed detection results
        display_dog_detection_results(detection_results)

    finally:
        # Clean up temp file
        os.remove(tmp_path)


if __name__ == "__main__":
    main()
