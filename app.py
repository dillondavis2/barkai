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

# Hugging Face model repository for breed classifier
HF_REPO_ID = "dllndvs/dogspeak-breed-classifier"
HF_MODEL_FILENAME = "random_forest_model.joblib"
LOCAL_MODEL_CACHE = Path("./model_cache")

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
def load_breed_classifier():
    # Download and load trained random forest model from Hugging Face Hub
    try:
        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_MODEL_FILENAME,
            cache_dir=LOCAL_MODEL_CACHE,
        )
        return joblib.load(model_path)
    except Exception as e:
        st.warning(f"Could not load breed classifier: {e}")
        return None

dog_detector = load_dog_detector()
breed_classifier = load_breed_classifier()

# File Uploader
audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "ogg"])

if audio_file is not None:
    # Display audio player
    st.audio(audio_file)

    # Save uploaded file to a temporary file (Transformers pipeline needs a path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp_file:
        tmp_file.write(audio_file.getvalue())
        tmp_path = tmp_file.name

    # Run Dog Detection
    with st.spinner("Listening for barks..."):
        results = dog_detector(tmp_path)

    # Logic: Check if "Dog" or "Bark" is in the top 5 predictions
    # AudioSet labels include 'Dog', 'Bark', 'Growling', etc.
    top_labels = [result['label'] for result in results[:5]]
    scores = {result['label']: result['score'] for result in results[:5]}

    # Check for dog-related keywords
    dog_detected = any(keyword in label for label in top_labels for keyword in ['Dog', 'Bark', 'Puppy'])

    if dog_detected:
        dog_confidence = scores.get('Dog', scores.get('Bark', 0.0))
        st.success(f"✅ **Dog Detected!** (Confidence: {dog_confidence:.2%})")

        # Run breed classification if dog detected and model is loaded
        if breed_classifier is not None:
            with st.spinner("Identifying breed..."):
                features = extract_mfcc_features(tmp_path)

                if features is not None:
                    # Get prediction and probabilities
                    breed_pred = breed_classifier.predict(features)[0]
                    breed_proba = breed_classifier.predict_proba(features)[0]
                    breed_classes = breed_classifier.classes_

                    # Format breed name nicely
                    breed_display = breed_pred.replace('_', ' ').title()
                    breed_confidence = max(breed_proba)

                    st.info(f"🐕 **Predicted Breed: {breed_display}** (Confidence: {breed_confidence:.2%})")

                    # Show breed probabilities
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
            st.warning("Breed classifier not available.")
    else:
        st.error("🚫 **No Dog Detected.**")

    # Clean up temp file
    os.remove(tmp_path)

    # Show raw detection results
    with st.expander("See detailed dog detection analysis"):
        st.write(results)

