import streamlit as st
import tempfile
import os
import numpy as np
import librosa
import joblib
import logging
from pathlib import Path
from huggingface_hub import hf_hub_download, InferenceClient

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

# --- Inference Client ---

@st.cache_resource
def get_inference_client():
    """Get HuggingFace Inference Client (uses HF_TOKEN from secrets if available)."""
    try:
        token = st.secrets.get("HF_TOKEN", None)
        logger.info(f"HF_TOKEN loaded: {'Yes' if token else 'No'}")
        if token:
            logger.info(f"Token prefix: {token[:10]}...")
    except Exception as e:
        logger.warning(f"Could not read secrets: {e}")
        token = None
    client = InferenceClient(token=token)
    logger.info(f"InferenceClient created: {client}")
    return client


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


# --- Local Model Loaders (Random Forest only) ---

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


# --- Dog Detection (via Inference API) ---

def detect_dog_via_api(audio_path, client, max_retries=3):
    """
    Run dog detection on audio file via HuggingFace Inference API.
    Returns (is_detected, confidence, full_results)
    Includes retry logic for model cold starts.
    """
    import time
    import traceback
    from huggingface_hub.utils import HfHubHTTPError

    logger.info(f"Starting dog detection for: {audio_path}")

    try:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        logger.info(f"Read audio file: {len(audio_bytes)} bytes")
    except Exception as e:
        logger.error(f"Failed to read audio file: {e}")
        st.error(f"Failed to read audio file: {e}")
        return False, 0.0, []

    last_error = None
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries}: Calling audio_classification API")
            logger.info(f"Model: {HF_DOG_DETECTOR_ID}")

            response = client.audio_classification(
                audio=audio_bytes,
                model=HF_DOG_DETECTOR_ID,
            )

            logger.info(f"API response received. Type: {type(response)}")
            logger.info(f"API response repr: {repr(response)}")

            # Ensure response is a list and handle empty/None responses
            if response is None:
                logger.warning("Response is None")
                st.warning("Dog detection returned no results")
                return False, 0.0, []

            # Convert to list if it's a generator/iterator
            logger.info(f"Converting response to list. Is list: {isinstance(response, list)}")
            try:
                response_list = list(response) if not isinstance(response, list) else response
                logger.info(f"Response list length: {len(response_list)}")
                if response_list:
                    logger.info(f"First item type: {type(response_list[0])}")
                    logger.info(f"First item repr: {repr(response_list[0])}")
            except Exception as conv_error:
                logger.error(f"Error converting response to list: {conv_error}")
                logger.error(traceback.format_exc())
                st.error(f"Error processing API response: {conv_error}")
                return False, 0.0, []

            if not response_list:
                logger.warning("Response list is empty")
                st.warning("Dog detection returned empty results")
                return False, 0.0, []

            # Convert API response to consistent format
            results = []
            for i, r in enumerate(response_list):
                logger.debug(f"Processing item {i}: type={type(r)}, repr={repr(r)}")
                if hasattr(r, 'label') and hasattr(r, 'score'):
                    results.append({"label": r.label, "score": r.score})
                elif isinstance(r, dict):
                    results.append({"label": r.get('label', ''), "score": r.get('score', 0.0)})
                else:
                    logger.warning(f"Unknown item format at index {i}: {type(r)}")

            logger.info(f"Parsed {len(results)} results")

            if not results:
                logger.warning("No results after parsing")
                st.warning("Could not parse dog detection results")
                return False, 0.0, []

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

        except HfHubHTTPError as e:
            last_error = e
            error_str = str(e)
            logger.error(f"HfHubHTTPError: {error_str}")
            logger.error(traceback.format_exc())
            # Model is loading (cold start) - wait and retry
            if "loading" in error_str.lower() or "503" in error_str:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10  # 10s, 20s, 30s
                    logger.info(f"Model loading, waiting {wait_time}s before retry")
                    st.info(f"Model is warming up... retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
            # Other HTTP error - don't retry
            st.error(f"Dog detection API error: {type(e).__name__}: {e}")
            return False, 0.0, []

        except Exception as e:
            last_error = e
            logger.error(f"Unexpected error: {type(e).__name__}: {e}")
            logger.error(traceback.format_exc())
            st.error(f"Dog detection error: {type(e).__name__}: {e}")
            return False, 0.0, []

    logger.error(f"All {max_retries} attempts failed. Last error: {last_error}")
    st.error(f"Dog detection failed after {max_retries} attempts: {last_error}")
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
    Classify breed using Random Forest model with MFCC features (local).
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


def classify_breed_with_ast_api(audio_path, model_id, client, max_retries=3):
    """
    Classify breed using AST model via HuggingFace Inference API.
    Returns (predicted_breed, confidence, results_list) or (None, None, None) on failure.
    Includes retry logic for model cold starts.
    """
    import time
    from huggingface_hub.utils import HfHubHTTPError

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.audio_classification(
                audio=audio_bytes,
                model=model_id,
            )

            # Ensure response is a list and handle empty/None responses
            if response is None:
                return None, None, None

            # Convert to list if it's a generator/iterator
            response_list = list(response) if not isinstance(response, list) else response

            if not response_list:
                return None, None, None

            # Convert API response to consistent format
            results = []
            for r in response_list:
                if hasattr(r, 'label') and hasattr(r, 'score'):
                    results.append({"label": r.label, "score": r.score})
                elif isinstance(r, dict):
                    results.append({"label": r.get('label', ''), "score": r.get('score', 0.0)})

            if not results:
                return None, None, None

            breed_pred = results[0]['label']
            confidence = results[0]['score']
            sorted_results = [(r['label'], r['score']) for r in results]

            return breed_pred, confidence, sorted_results

        except HfHubHTTPError as e:
            last_error = e
            error_str = str(e)
            # Model is loading (cold start) - wait and retry
            if "loading" in error_str.lower() or "503" in error_str:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10
                    st.info(f"Model is warming up... retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
            st.error(f"Breed classification API error: {type(e).__name__}: {e}")
            return None, None, None

        except Exception as e:
            last_error = e
            st.error(f"Breed classification error: {type(e).__name__}: {e}")
            return None, None, None

    st.error(f"Breed classification failed after {max_retries} attempts: {last_error}")
    return None, None, None


def display_breed_results(breed_pred, confidence, sorted_results):
    """Display breed classification results with progress bars."""
    breed_display = breed_pred.replace('_', ' ').title()
    st.info(f"🐕 **Predicted Breed: {breed_display}** (Confidence: {confidence:.2%})")

    with st.expander("See breed probabilities"):
        for breed, prob in sorted_results:
            breed_name = breed.replace('_', ' ').title()
            st.write(f"**{breed_name}:** {prob:.2%}")
            st.progress(prob)


def run_breed_classification(audio_path, model_type, model_name, client):
    """Run breed classification based on selected model type."""
    breed_pred, confidence, sorted_results = None, None, None

    if model_type == "random_forest":
        classifier = load_rf_breed_classifier()
        if classifier is None:
            st.warning(f"{model_name} not available.")
            return
        breed_pred, confidence, sorted_results = classify_breed_with_rf(audio_path, classifier)

    elif model_type == "youtube_random_forest":
        classifier = load_yt_rf_breed_classifier()
        if classifier is None:
            st.warning(f"{model_name} not available.")
            return
        breed_pred, confidence, sorted_results = classify_breed_with_rf(audio_path, classifier)

    elif model_type == "ast":
        breed_pred, confidence, sorted_results = classify_breed_with_ast_api(
            audio_path, HF_AST_REPO_ID, client
        )

    elif model_type == "youtube_ast":
        breed_pred, confidence, sorted_results = classify_breed_with_ast_api(
            audio_path, HF_YT_AST_REPO_ID, client
        )

    if breed_pred is None:
        st.warning("Could not classify breed.")
        return

    display_breed_results(breed_pred, confidence, sorted_results)


# --- Main App ---

def main():
    st.title("🐶 Dog Bark Detector & Breed Classifier")
    st.write("Upload an audio file to check if it contains a dog barking and identify the breed.")

    # Get inference client
    client = get_inference_client()

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
        # Dog detection via API
        with st.spinner("Listening for barks..."):
            is_detected, dog_confidence, detection_results = detect_dog_via_api(tmp_path, client)

        if is_detected:
            st.success(f"✅ **Dog Detected!** (Confidence: {dog_confidence:.2%})")

            # Breed classification
            with st.spinner(f"Identifying breed using {selected_model_name}..."):
                run_breed_classification(tmp_path, selected_model_type, selected_model_name, client)
        else:
            st.error("🚫 **No Dog Detected.**")

        # Show detailed detection results
        display_dog_detection_results(detection_results)

    finally:
        # Clean up temp file
        os.remove(tmp_path)


if __name__ == "__main__":
    main()
