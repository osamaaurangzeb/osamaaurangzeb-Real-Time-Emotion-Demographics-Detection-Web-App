from flask import Flask, request, jsonify, render_template, Blueprint
import numpy as np
import librosa
import os
import time
from tensorflow.keras.models import load_model
import sounddevice as sd
from scipy.io.wavfile import write
import glob

speech_bp = Blueprint("speech", __name__, static_folder='static', template_folder='templates')

# Load model once at startup
model = load_model('blueprints/speech/speech_model.h5')  

# Emotion mapping
emotion_map = {
    0: 'Neutral', 1: 'Calm', 2: 'Happy', 3: 'Sad',
    4: 'Angry', 5: 'Fearful', 6: 'Disgust', 7: 'Surprised'
}

latest_emotion = "N/A"  # Initialize global variable to hold the latest emotion

# Global variables for recording
recording = False
audio_data = []

def record_audio(duration=5, sample_rate=48000):
    """Records audio for a specified duration and saves it with a timestamp."""
    global recording, audio_data
    audio_data = []  # Clear previous audio data
    recording = True
    print("Recording...")

    def callback(indata, frames, time, status):
        if recording:
            audio_data.append(indata.copy())

    with sd.InputStream(callback=callback, channels=1, samplerate=sample_rate):
        while recording:
            sd.sleep(100)

    # Save as WAV with a unique name using the current timestamp
    audio_filename = os.path.join('blueprints/speech/static', f'temp_audio.wav')
    write(audio_filename, sample_rate, (np.concatenate(audio_data) * 32767).astype(np.int16))
    print(f"Recording complete. File saved as '{audio_filename}'.")

@speech_bp.route('/')
def index():
    return render_template('index.html')

@speech_bp.route('/start_recording', methods=['POST'])
def start_recording():
    global recording
    if not recording:
        # Delete previous recording if it exists
        previous_file = 'static/temp_audio.wav'
        if os.path.exists(previous_file):
            os.remove(previous_file)
            print(f"Deleted previous recording: {previous_file}")
        
        record_audio()  # Start recording
    return jsonify({'status': 'Processing...'})

@speech_bp.route('/stop_recording', methods=['POST'])
def stop_recording():
    global recording
    recording = False  # Stop recording
    return jsonify({'status': 'Recording stopped'})

@speech_bp.route('/predict_emotion', methods=['POST'])
def predict_emotion():
    global latest_emotion

    time.sleep(0.5)  # Brief pause to ensure file writing completes

    audio_filename = 'blueprints/speech/static/temp_audio.wav'
    if not os.path.exists(audio_filename):
        return jsonify({'error': 'Audio file not found.'}), 404

    try:
        # Load and process audio for prediction
        y, sr = librosa.load(audio_filename, sr=None)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        x = np.expand_dims(np.expand_dims(mfccs, axis=1), axis=0)

        # Predict emotion
        predictions = model.predict(x)
        predicted_class = np.argmax(predictions, axis=1)[0]
        detected_emotion = emotion_map.get(predicted_class, 'unknown')

        latest_emotion = detected_emotion

        # Delete the audio file after prediction
        # os.remove(audio_filename)
        # print(f"Deleted '{audio_filename}' after prediction.")

        return jsonify({'emotion': latest_emotion})

    except Exception as e:
        print("Error during processing:", str(e))
        return jsonify({'error': str(e)}), 500
    