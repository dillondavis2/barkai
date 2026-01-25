import streamlit as st
from transformers import pipeline
import tempfile
import os

# Title and Instructions
st.title("🐶 Dog Bark Detector")
st.write("Upload an audio file to check if it contains a dog barking.")

# Load the model (Cached so it doesn't reload on every click)
@st.cache_resource
def load_model():
    # Using MIT's AST model fine-tuned on AudioSet (includes 'Dog', 'Bark' classes)
    return pipeline("audio-classification", model="MIT/ast-finetuned-audioset-10-10-0.4593")

model = load_model()

# File Uploader
audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "ogg"])

if audio_file is not None:
    # Display audio player
    st.audio(audio_file)

    # Save uploaded file to a temporary file (Transformers pipeline needs a path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp_file:
        tmp_file.write(audio_file.getvalue())
        tmp_path = tmp_file.name

    # Run Inference
    with st.spinner("Listening for barks..."):
        results = model(tmp_path)
        
        # Clean up temp file
        os.remove(tmp_path)

    # Logic: Check if "Dog" or "Bark" is in the top 5 predictions
    # AudioSet labels include 'Dog', 'Bark', 'Growling', etc.
    top_labels = [result['label'] for result in results[:5]]
    scores = {result['label']: result['score'] for result in results[:5]}
    
    # Check for dog-related keywords
    dog_detected = any(keyword in label for label in top_labels for keyword in ['Dog', 'Bark', 'Puppy'])

    if dog_detected:
        st.success(f"✅ **Dog Detected!** (Confidence: {scores.get('Dog', scores.get('Bark', 0.0)):.2%})")
    else:
        st.error("🚫 **No Dog Detected.**")
        
    # Show raw results for the interview (shows you care about transparency)
    with st.expander("See detailed analysis"):
        st.write(results)
  
