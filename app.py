import streamlit as st
from transformers import pipeline
import tempfile
import os
import numpy as np
import librosa
import joblib
from pathlib import Path
from huggingface_hub import hf_hub_download

# Title and Instructions
st.title("🐶 Dog Bark Detector & Breed Classifier")
st.write("Upload an audio file to check if it contains a dog barking and identify the breed.")

# Hugging Face model repositories for breed classifiers
HF_RF_REPO_ID = "dllndvs/dogspeak-breed-classifier"
HF_RF_MODEL_FILENAME = "random_forest_model.joblib"
HF_AST_REPO_ID = "dllndvs/dogspeak-ast-breed-classifier"
LOCAL_MODEL_CACHE = Path("./model_cache")

# Model selection options
BREED_MODEL_OPTIONS = {
    "Random Forest (MFCC features)": "random_forest",
    "Dogspeak AST (Fine-tuned)": "ast"
}

def extract_mfcc_features(file_path):
    """
    Extract MFCC features from audio file (same as training).
    Returns 40-dim vector: 20 MFCC means + 20 MFCC stds
    """
    try:
        # Load audio (resample to 16k for consistency with training)
        y, sr = librosa.load(file_path, sr=16000)

        # Remove silence
        y, _ = librosa.effects.trim(y)

        # Extract MFCCs (n_mfcc=20)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

        # Collapse time: Mean and Std of each MFCC band
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        # Concatenate to get 40-dim vector
        return np.concatenate([mfcc_mean, mfcc_std]).reshape(1, -1)
    except Exception as e:
        st.warning(f"Error extracting features: {e}")
        return None

# Load models (Cached so they don't reload on every click)
@st.cache_resource
def load_dog_detector():
    # Using MIT's AST model fine-tuned on AudioSet (includes 'Dog', 'Bark' classes)
    return pipeline("audio-classification", model="MIT/ast-finetuned-audioset-10-10-0.4593")

@st.cache_resource
def load_rf_breed_classifier():
    # Download and load trained random forest model from Hugging Face Hub
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
    # Load fine-tuned AST model for breed classification
    try:
        return pipeline("audio-classification", model=HF_AST_REPO_ID)
    except Exception as e:
        st.warning(f"Could not load AST breed classifier: {e}")
        return None

dog_detector = load_dog_detector()

# Model selection dropdown
selected_model_name = st.selectbox(
    "Select Breed Classification Model",
    options=list(BREED_MODEL_OPTIONS.keys()),
    index=0  # Default to Random Forest
)
selected_model_type = BREED_MODEL_OPTIONS[selected_model_name]

# File Uploader
audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "ogg"])

if audio_file is not None:
    # Display audio player
    st.audio(audio_file)

    # Save uploaded file to a temporary file (Transformers pipeline needs a path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp_file:
        tmp_file.write(audio_file.getvalue())
        tmp_path = tmp_file.name

    # Run Dog Detection (get top 100 for detailed analysis)
    with st.spinner("Listening for barks..."):
        results = dog_detector(tmp_path, top_k=100)

    # Logic: Check if "Dog" or "Bark" is in the top 5 predictions
    # AudioSet labels include 'Dog', 'Bark', 'Growling', etc.
    top_labels = [result['label'] for result in results[:5]]
    scores = {result['label']: result['score'] for result in results[:5]}

    # Check for dog-related keywords
    dog_detected = any(keyword in label for label in top_labels for keyword in ['Dog', 'Bark', 'Puppy'])

    if dog_detected:
        dog_confidence = scores.get('Dog', scores.get('Bark', 0.0))
        st.success(f"✅ **Dog Detected!** (Confidence: {dog_confidence:.2%})")

        # Run breed classification based on selected model
        with st.spinner(f"Identifying breed using {selected_model_name}..."):
            if selected_model_type == "random_forest":
                # Load and run Random Forest model
                rf_classifier = load_rf_breed_classifier()
                if rf_classifier is not None:
                    features = extract_mfcc_features(tmp_path)
                    if features is not None:
                        breed_pred = rf_classifier.predict(features)[0]
                        breed_proba = rf_classifier.predict_proba(features)[0]
                        breed_classes = rf_classifier.classes_

                        breed_display = breed_pred.replace('_', ' ').title()
                        breed_confidence = max(breed_proba)

                        st.info(f"🐕 **Predicted Breed: {breed_display}** (Confidence: {breed_confidence:.2%})")

                        with st.expander("See breed probabilities"):
                            breed_results = sorted(
                                zip(breed_classes, breed_proba),
                                key=lambda x: x[1],
                                reverse=True
                            )
                            for breed, prob in breed_results:
                                breed_name = breed.replace('_', ' ').title()
                                st.write(f"**{breed_name}:** {prob:.2%}")
                                st.progress(prob)
                else:
                    st.warning("Random Forest breed classifier not available.")

            elif selected_model_type == "ast":
                # Load and run AST model
                ast_classifier = load_ast_breed_classifier()
                if ast_classifier is not None:
                    ast_results = ast_classifier(tmp_path, top_k=100)

                    breed_pred = ast_results[0]['label']
                    breed_confidence = ast_results[0]['score']
                    breed_display = breed_pred.replace('_', ' ').title()

                    st.info(f"🐕 **Predicted Breed: {breed_display}** (Confidence: {breed_confidence:.2%})")

                    with st.expander("See breed probabilities"):
                        for result in ast_results:
                            breed_name = result['label'].replace('_', ' ').title()
                            prob = result['score']
                            st.write(f"**{breed_name}:** {prob:.2%}")
                            st.progress(prob)
                else:
                    st.warning("AST breed classifier not available.")
    else:
        st.error("🚫 **No Dog Detected.**")

    # Clean up temp file
    os.remove(tmp_path)

    # Show detailed detection results as bar graph
    with st.expander("See detailed dog detection analysis"):
        for result in results:
            label = result['label']
            score = result['score']
            st.write(f"**{label}:** {score:.2%}")
            st.progress(score)

