## Violin Emotion Analysis
A deep learning system for analyzing emotional content in violin audio recordings using temporal feature extraction and hybrid LSTM-Random Forest modeling.

## Overview
This project implements a machine learning pipeline that:

Processes violin audio recordings to extract rich temporal and spectral features

Uses a hybrid LSTM + Random Forest model to predict emotional content

Provides explainable AI insights through SHAP analysis

Supports soft emotion labels (probability distributions across multiple emotions)

Features
Audio Processing: Noise reduction, normalization, and temporal feature extraction

Data Augmentation: Pitch shifting and time stretching for robust training

Hybrid Modeling: Combines LSTM temporal modeling with Random Forest regression

Explainable AI: SHAP analysis for feature importance interpretation

Multi-emotion Support: Handles soft labels with probability distributions

## Installation:

pip install librosa scikit-learn matplotlib numpy pandas noisereduce tensorflow shap soundfile tqdm joblib

## Project Structure

violin-emotion-analysis/

├── data/                    # Processed audio files

├── models/                  # Saved model files

├── emotion_labels.csv       # Emotion annotations with soft labels

├── hybrid_model/           # Trained hybrid model components

│   ├── lstm_model.h5

│   ├── rf_model.pkl

│   └── meta.json

## Data Requirements
Audio Files Format: WAV files

Sample rate: 44.1 kHz

Location: /kaggle/input/train-audio/audio/ or specified directory

Emotion Labels CSV
Required columns:

filename: Name of the audio file

emotion: Primary emotion label

Additional columns for soft emotion probabilities (e.g., happy, sad, angry)

Example:

csv

filename,emotion,happy,sad,angry

violin1.wav,happy,0.8,0.1,0.1

violin2.wav,sad,0.2,0.7,0.1

## Usage
1. Data Preparation

 Set your data paths
data_dir = '/path/to/audio/files'
emotion_csv_path = '/path/to/emotion_labels.csv'

Create temporal dataset with augmentation
X, y_soft = create_temporal_dataset(data_dir, emotion_csv_path, augment=True)

2. Model Training

The system automatically trains multiple runs and selects the best model
Training includes:
- LSTM for temporal sequence modeling
- Random Forest on LSTM embeddings
- Weighted fusion of predictions

3. Model Inference

Load trained model

hybrid_model = HybridEmotionModel.load('hybrid_model')

Make predictions
predictions = hybrid_model.predict(X_test)

4. Explainability

- Generate SHAP analysis for feature importance
- Shows which audio features contribute to each emotion prediction

## Feature Extraction
The system extracts comprehensive audio features:

Spectral Features: MFCCs, chroma, spectral contrast

Temporal Features: RMS energy, zero-crossing rate

Expression Features: Pitch, tonality, articulation

Statistical Summaries: Mean, standard deviation, temporal curves

## Model Architecture
Hybrid Emotion Model
LSTM Component: Bidirectional LSTM with layer normalization

Captures temporal dependencies in audio sequences

Outputs 64-dimensional embeddings

Random Forest Component:

Takes LSTM embeddings as input

Provides robust regression on emotion probabilities

Fusion: Weighted combination (70% LSTM + 30% RF)

## Output
The model predicts probability distributions across emotion categories.
(Customizable based on your label set)

## Performance Metrics
R² Score: Measures prediction accuracy for soft labels

Cross-validation: Multiple training runs with best model selection

Early Stopping: Prevents overfitting during LSTM training

## Customization
Adding New Emotions:
Update the emotion labels in your CSV file

Add new columns for soft label probabilities

The model automatically adapts to the output dimension


## Dependencies
Audio Processing: librosa, noisereduce, soundfile

Machine Learning: scikit-learn, tensorflow, shap

Utilities: numpy, pandas, matplotlib, joblib

## Notes
The system is optimized for violin audio but can be adapted for other instruments

Default parameters work well for 3-5 second audio clips

For longer recordings, adjust window_size and hop_time parameters

GPU acceleration recommended for LSTM training
