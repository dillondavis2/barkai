import streamlit as st
import tempfile
import os
import gc
import numpy as np
import librosa
import joblib
import logging
from pathlib import Path
from huggingface_hub import hf_hub_download

# Configure logging
logging.basicConfig(
    level=logging.INFO,
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

# File limits
MAX_FILE_SIZE_MB = 10  # Maximum upload size in MB
MAX_AUDIO_DURATION_SEC = 60  # Maximum audio duration in seconds
TARGET_SAMPLE_RATE = 16000  # Sample rate for processing


def clear_memory():
    """Aggressively clear memory."""
    gc.collect()
    # Only import torch if needed, and clear its cache
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# --- Audio Processing ---

def validate_audio_file(audio_file):
    """
    Validate uploaded audio file size.
    Returns (is_valid, error_message).
    """
    # Check file size
    file_size_mb = len(audio_file.getvalue()) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        return False, f"File too large: {file_size_mb:.1f}MB (max {MAX_FILE_SIZE_MB}MB)"
    return True, None


def validate_audio_duration(file_path):
    """
    Validate audio duration after saving to temp file.
    Returns (is_valid, duration_sec, error_message).
    """
    try:
        # Use librosa to get duration without loading entire file
        duration = librosa.get_duration(path=file_path)
        if duration > MAX_AUDIO_DURATION_SEC:
            return False, duration, f"Audio too long: {duration:.1f}s (max {MAX_AUDIO_DURATION_SEC}s)"
        if duration < 0.5:
            return False, duration, f"Audio too short: {duration:.2f}s (min 0.5s)"
        return True, duration, None
    except Exception as e:
        return False, 0, f"Could not read audio file: {e}"


def save_audio_to_temp(audio_file):
    """Save uploaded audio file to a temporary file and return the path."""
    suffix = os.path.splitext(audio_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(audio_file.getvalue())
        return tmp_file.name


def load_and_trim_audio(file_path, max_duration=None):
    """
    Load audio file, optionally trimming to max duration.
    Returns (audio_array, sample_rate) or (None, None) on error.
    """
    try:
        # Load with target sample rate
        y, sr = librosa.load(file_path, sr=TARGET_SAMPLE_RATE)

        # Trim to max duration if specified
        if max_duration is not None:
            max_samples = int(max_duration * sr)
            if len(y) > max_samples:
                y = y[:max_samples]

        return y, sr
    except Exception as e:
        logger.error(f"Error loading audio: {e}")
        return None, None


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


# --- Unified Model Manager ---
# Only ONE model (either dog detector OR breed classifier) is in memory at a time
# This is critical for staying within memory limits

class ModelManager:
    """
    Manages ALL model loading/unloading to prevent memory exhaustion.
    Only ONE model is kept in memory at a time (dog detector OR breed classifier).
    """

    def __init__(self):
        self._current_model = None
        self._current_model_id = None

    def _unload_current_model(self):
        """Unload the currently loaded model and free memory."""
        if self._current_model is not None:
            logger.info(f"Unloading model: {self._current_model_id}")
            del self._current_model
            self._current_model = None
            self._current_model_id = None
            clear_memory()

    def get_dog_detector(self):
        """Load and return the dog detector model."""
        model_id = "dog_detector"

        if self._current_model_id == model_id:
            return self._current_model

        # Unload any current model first
        self._unload_current_model()

        logger.info("Loading dog detector model...")
        try:
            # Import transformers only when needed
            from transformers import pipeline
            model = pipeline("audio-classification", model=HF_DOG_DETECTOR_ID)
            self._current_model = model
            self._current_model_id = model_id
            logger.info("Dog detector loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Failed to load dog detector: {e}")
            st.error(f"Failed to load dog detector: {e}")
            return None

    def get_breed_classifier(self, model_type):
        """Load and return a breed classifier."""
        model_id = f"breed_{model_type}"

        if self._current_model_id == model_id:
            return self._current_model

        # Unload any current model first
        self._unload_current_model()

        logger.info(f"Loading breed classifier: {model_type}")
        try:
            if model_type == "random_forest":
                model = self._load_rf_model(HF_RF_REPO_ID, HF_RF_MODEL_FILENAME)
            elif model_type == "youtube_random_forest":
                model = self._load_rf_model(HF_YT_RF_REPO_ID, HF_YT_RF_MODEL_FILENAME)
            elif model_type == "ast":
                model = self._load_ast_model(HF_AST_REPO_ID)
            elif model_type == "youtube_ast":
                model = self._load_ast_model(HF_YT_AST_REPO_ID)
            else:
                logger.error(f"Unknown model type: {model_type}")
                return None

            self._current_model = model
            self._current_model_id = model_id
            logger.info(f"Breed classifier loaded: {model_type}")
            return model

        except Exception as e:
            logger.error(f"Failed to load breed classifier: {e}")
            st.error(f"Failed to load breed classifier: {e}")
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
        from transformers import pipeline
        return pipeline("audio-classification", model=repo_id)

    def unload_all(self):
        """Explicitly unload all models."""
        self._unload_current_model()


def get_model_manager():
    """Get or create the model manager in session state."""
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = ModelManager()
    return st.session_state.model_manager


# --- Dog Detection ---

def detect_dog(audio_path, model_manager):
    """
    Run dog detection on audio file.
    Returns (is_detected, confidence, top_results)
    """
    logger.info(f"Starting dog detection for: {audio_path}")

    detector = model_manager.get_dog_detector()
    if detector is None:
        return False, 0.0, []

    try:
        # Only get top 10 to reduce memory
        results = detector(audio_path, top_k=10)
        logger.info(f"Dog detection returned {len(results)} results")

        top_labels = [result['label'] for result in results[:5]]
        scores = {result['label']: result['score'] for result in results[:5]}

        is_detected = any(
            keyword in label
            for label in top_labels
            for keyword in ['Dog', 'Bark', 'Puppy']
        )
        confidence = scores.get('Dog', scores.get('Bark', 0.0))

        return is_detected, confidence, results[:5]

    except Exception as e:
        logger.error(f"Dog detection error: {e}")
        st.error(f"Dog detection error: {e}")
        return False, 0.0, []


def display_dog_detection_results(results):
    """Display detailed dog detection analysis."""
    if not results:
        return
    with st.expander("See dog detection details"):
        for result in results:
            st.write(f"**{result['label']}:** {result['score']:.2%}")
            st.progress(result['score'])


# --- Breed Classification ---

def classify_breed_with_rf(audio_path, classifier):
    """Classify breed using Random Forest model with MFCC features."""
    features = extract_mfcc_features(audio_path)
    if features is None:
        return None, None, None

    breed_pred = classifier.predict(features)[0]
    breed_proba = classifier.predict_proba(features)[0]
    breed_classes = classifier.classes_

    # Only keep top 10 results to save memory
    sorted_results = sorted(
        zip(breed_classes, breed_proba),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    return breed_pred, max(breed_proba), sorted_results


def classify_breed_with_ast(audio_path, classifier):
    """Classify breed using AST model."""
    try:
        # Only get top 10 to reduce memory
        results = classifier(audio_path, top_k=10)

        if not results:
            return None, None, None

        breed_pred = results[0]['label']
        confidence = results[0]['score']
        sorted_results = [(r['label'], r['score']) for r in results]

        return breed_pred, confidence, sorted_results

    except Exception as e:
        logger.error(f"AST breed classification error: {e}")
        st.error(f"Breed classification error: {e}")
        return None, None, None


def display_breed_results(breed_pred, confidence, sorted_results):
    """Display breed classification results."""
    breed_display = breed_pred.replace('_', ' ').title()
    st.info(f"**Predicted Breed: {breed_display}** (Confidence: {confidence:.2%})")

    with st.expander("See breed probabilities"):
        for breed, prob in sorted_results:
            breed_name = breed.replace('_', ' ').title()
            st.write(f"**{breed_name}:** {prob:.2%}")
            st.progress(prob)


def run_breed_classification(audio_path, model_type, model_name, model_manager):
    """Run breed classification based on selected model type."""
    classifier = model_manager.get_breed_classifier(model_type)

    if classifier is None:
        st.warning(f"{model_name} not available.")
        return

    if model_type in ["random_forest", "youtube_random_forest"]:
        breed_pred, confidence, sorted_results = classify_breed_with_rf(audio_path, classifier)
    else:
        breed_pred, confidence, sorted_results = classify_breed_with_ast(audio_path, classifier)

    if breed_pred is None:
        st.warning("Could not classify breed.")
        return

    display_breed_results(breed_pred, confidence, sorted_results)


# --- Main App ---

def main():
    st.set_page_config(
        page_title="Dog Bark Classifier",
        page_icon="🐕",
        layout="centered"
    )

    st.title("Dog Bark Detector & Breed Classifier")
    st.write("Upload an audio file to check if it contains a dog barking and identify the breed.")
    st.caption("Memory optimized: Models are loaded on-demand and only one at a time.")

    # Model selection dropdown
    selected_model_name = st.selectbox(
        "Select Breed Classification Model",
        options=list(BREED_MODEL_OPTIONS.keys()),
        index=0
    )
    selected_model_type = BREED_MODEL_OPTIONS[selected_model_name]

    # File uploader with size limit info
    st.caption(f"Max file size: {MAX_FILE_SIZE_MB}MB | Max duration: {MAX_AUDIO_DURATION_SEC}s")
    audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "ogg"])

    if audio_file is None:
        st.info("Please upload an audio file to begin.")
        return

    # Step 1: Validate file size (before saving to disk)
    is_valid, error_msg = validate_audio_file(audio_file)
    if not is_valid:
        st.error(f"**{error_msg}**")
        st.info("Please upload a smaller file. For long recordings, trim to the section with barking.")
        return

    # Display audio player
    st.audio(audio_file)

    # Save to temp file
    tmp_path = save_audio_to_temp(audio_file)

    try:
        # Step 2: Validate audio duration
        is_valid, duration, error_msg = validate_audio_duration(tmp_path)
        if not is_valid:
            st.error(f"**{error_msg}**")
            st.info("Please upload a shorter audio clip containing dog barking.")
            return

        st.caption(f"Audio duration: {duration:.1f}s")

        # Get model manager
        model_manager = get_model_manager()

        # Step 3: Dog detection
        with st.spinner("Listening for barks..."):
            is_detected, dog_confidence, detection_results = detect_dog(tmp_path, model_manager)

        if not is_detected:
            st.error("**No Dog Detected.**")
            display_dog_detection_results(detection_results)
            return

        st.success(f"**Dog Detected!** (Confidence: {dog_confidence:.2%})")
        display_dog_detection_results(detection_results)

        # Step 4: Breed classification
        # Note: This will unload the dog detector to free memory
        with st.spinner(f"Identifying breed using {selected_model_name}..."):
            run_breed_classification(tmp_path, selected_model_type, selected_model_name, model_manager)

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        st.error(f"An error occurred: {e}")

    finally:
        # Clean up temp file
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass  # Ignore cleanup errors


if __name__ == "__main__":
    main()
