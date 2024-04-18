import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import librosa
from skimage.transform import resize

# Function to preprocess audio file into spectrogram
def preprocess_audio(audio_file):
    y, sr = librosa.load(audio_file)
    D = librosa.stft(y)
    D_db = librosa.amplitude_to_db(abs(D))
    resized_spectrogram = resize(D_db, (256, 256), anti_aliasing=True)
    resized_spectrogram = np.expand_dims(resized_spectrogram, axis=-1)
    return resized_spectrogram

# Load the trained model
model = load_model('/content/drive/MyDrive/audio.h5')

# Streamlit app
st.title('Audio Classification')
st.sidebar.title('Upload Audio File')

# Upload audio file
uploaded_file = st.sidebar.file_uploader("Choose an audio file", type=['wav'])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    # Preprocess the audio file
    spectrogram = preprocess_audio(uploaded_file)

    # Perform prediction
    prediction = model.predict(np.expand_dims(spectrogram, axis=0))
    class_names = ['Fake', 'Real']
    predicted_class = class_names[np.argmax(prediction)]

    # Display prediction
    st.write('Predicted Class:', predicted_class)
