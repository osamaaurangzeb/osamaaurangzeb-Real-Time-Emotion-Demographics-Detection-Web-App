﻿# Face, Age-Gender & Speech Emotion Recognition Web App

A multi-modal AI-powered web application for real-time facial emotion detection, age and gender prediction, and speech-based emotion recognition. Built with Flask, OpenCV, Keras/TensorFlow, and deep learning models.

## Features

- **Facial Emotion Recognition:** Detects emotions (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise) from live webcam feed.
- **Age & Gender Prediction:** Predicts age and gender from detected faces in real time.
- **Speech Emotion Recognition:** Records audio and predicts the speaker's emotion (Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised).
- **Modular Flask Architecture:** Organized with blueprints for easy extension and maintenance.
- **Interactive Web UI:** User-friendly interface for all features.
  
# Deep Learning Models & Performance
1. Facial Emotion Recognition Model
Architecture: Convolutional Neural Network (CNN) trained on the FER2013 dataset.
Test Accuracy: 85.96%
Details: The model achieves high accuracy, outperforming many existing state-of-the-art solutions. It is robust to real-world webcam input and can distinguish between seven basic emotions.
2. Age & Gender Prediction Model
Architecture: CNN-based regression and classification model.
Dataset: Trained on a labeled dataset of facial images with age and gender annotations.
Performance: The model provides reliable age estimation and gender classification, with Mean Absolute Error (MAE) and classification accuracy evaluated during training (see notebook for details).
3. Speech Emotion Recognition Model
Architecture: Deep neural network trained on MFCC features extracted from audio (RAVDESS dataset).
Test Accuracy: 88.1%
Details: The model can classify eight different emotions from speech with high accuracy, making it suitable for real-time audio emotion analysis.

## Project Structure

```
Face_Emotions/
│
├── Age-Gender Model/
│   ├── Age_Gender_model.ipynb      # Jupyter notebook for age-gender model training
│   ├── age.h5                      # Trained Keras model for age-gender
│   └── age_pickle.pkl              # Model weights or preprocessing objects
│
├── Expression Model/
│   ├── Facial Emotion Detection & Recognition using CNN.ipynb  # Notebook for emotion model
│   ├── expression_emotion_model.h5                            # Trained Keras model
│   └── expression_emotion_model.keras                         # Model in Keras format
│
├── Speech Model/
│   ├── speech_model.ipynb          # Notebook for speech emotion model
│   ├── speech_model.h5             # Trained Keras model
│   └── speech_model_pickle.pkl     # Model weights or preprocessing objects
│
└── Flask WebAPP/
    ├── app.py                      # Main Flask app entry point
    └── blueprints/
        ├── home/                   # Home page blueprint
        ├── expression/             # Facial emotion & age-gender blueprint
        └── speech/                 # Speech emotion blueprint
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/osamaaurangzeb/osamaaurangzeb-Real-Time-Emotion-Demographics-Detection-Web-App.git
   cd Face_Emotions/ FlasWebAPP
   ```

2. **Install dependencies:**
   - Python 3.7+
   - [pip](https://pip.pypa.io/en/stable/)
   - Recommended: Create a virtual environment


3. **Run the application:**
   ```bash
   python app.py
   ```
   - The app will be available at `http://127.0.0.1:5000/`

## Usage

- **Home:** Navigate to `/home` for the landing page.
- **Facial Emotion & Age-Gender:** Go to `/expression-age` for real-time webcam-based detection.
- **Speech Emotion:** Go to `/speech` to record audio and detect emotion.

## Model Training

- Jupyter notebooks for training each model are provided in their respective folders.
- Datasets are not included due to size; see notebook instructions for dataset sources and preprocessing.

## File Descriptions

- `app.py`: Registers blueprints and runs the Flask server.
- `blueprints/expression/expression.py`: Handles webcam video streaming, face detection, emotion, and age-gender prediction.
- `blueprints/speech/speech.py`: Handles audio recording and speech emotion prediction.
- `blueprints/home/home.py`: Renders the home page.



## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements

- FER2013 dataset for facial emotion recognition
- RAVDESS dataset for speech emotion recognition
- Keras, TensorFlow, OpenCV, Flask, and other open-source libraries

