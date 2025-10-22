
## Overview
This project trains a **hybrid deep learning and machine learning system** to recognize emotions in violin performances.  
The notebook `emotion-detector-2.ipynb` automatically:
1. Preprocesses audio data (noise reduction and feature extraction),
2. Builds soft emotion label datasets,
3. Trains an LSTM-based neural network and a Random Forest classifier,
4. Combines their predictions into a hybrid model, and
5. Visualizes and explains emotion predictions for new audio clips.

All steps run end-to-end inside Kaggle — once you upload your data ZIP file and emotion labels CSV, the notebook does the rest automatically.


## How It Works

### Cell 1 – Setup and Installation
Installs all required libraries:
librosa, scikit-learn, matplotlib, numpy, pandas, noisereduce, tensorflow, shap

### Cell 1.5 – Unzip and Verify Data
- Unzips your uploaded ZIP file into `/kaggle/working/violin-emotion-analysis/data`.
- Prints the folder structure to confirm your audio files and CSV were extracted correctly.
- Checks that `emotion_labels.csv` exists in the data folder.

Your ZIP should contain:
/data
├── audio1.wav
├── audio2.wav
└── emotion_labels.csv


### Cell 2 – Imports and Feature Extraction
Defines the feature extraction and dataset creation functions:
- `extract_temporal_features()`:  
  Loads an audio file, applies **noise reduction**, and extracts:
  - MFCCs
  - Chroma
  - Spectral contrast
  - RMS energy
  - Zero-crossing rate
  - Spectral centroid
  - Bandwidth  
  Features are extracted from **overlapping 3-second windows** to preserve time-based emotional flow.

- `create_temporal_dataset()`:  
  Loads each `.wav` file, calls `extract_temporal_features()` on it, and builds the training dataset using **soft emotion labels** (e.g., probabilities across several emotions).

### Cell 3 – Data Preparation
- Loads your audio dataset from the `/data` folder and your `emotion_labels.csv`.
- Builds feature arrays (`X`) and corresponding soft label arrays (`y_soft`).
- Pads sequences so all clips have consistent dimensions for the LSTM.
- Prints dataset size and number of emotion categories detected.

Your CSV must include **soft labels** like:
filename,emotion,happy,sad,calm,angry
violin1.wav,happy,0.7,0.1,0.2,0.0
violin2.wav,sad,0.1,0.8,0.1,0.0


### Cell 4 – Model Training (Hybrid LSTM + Random Forest)
1. **Splits data** into 80% training / 20% testing.  
2. **Normalizes** feature values for each sequence using `StandardScaler`.  
3. **Trains an LSTM model** with:
   - Two stacked LSTM layers
   - Attention mechanism
   - Batch normalization
   - Dropout regularization
   - Softmax output for multiple emotions  
4. **Trains a Random Forest classifier** on averaged (non-sequential) features.  
5. **Combines predictions** using a weighted fusion:
   - 60% LSTM output
   - 40% Random Forest output  
6. Saves trained models:
   - `/models/lstm_hybrid_model.h5`
   - `/models/rf_model.pkl`
   - `/models/scaler.pkl`  
7. Prints hybrid model accuracy.

### Cell 5 – Explainability and Visualization
- Creates a SHAP explainer for the Random Forest model.  
- Generates a SHAP summary plot to visualize which features (e.g., MFCCs or spectral contrast) most strongly affect each emotion prediction.


### Cell 6 – Real-Time Prediction Simulation
1. Reads new `.wav` files placed in `/kaggle/working/violin-emotion-analysis/new_audio/`.
2. Extracts temporal features for each file.
3. Pads the sequence and runs the trained **LSTM** and **Random Forest** models.
4. Combines their predictions (same 0.6 / 0.4 weighting).
5. Applies perceptual smoothing (3-frame moving average).
6. Displays a bar graph showing probabilities for each emotion.
7. Prints each emotion’s probability and the final predicted label.

Output example:
Audio: test_clip.wav → Predicted Emotion: Calm
Probability per emotion:
happy: 0.183
sad: 0.052
calm: 0.734
angry: 0.031
markdown
Copy code

## How to Use This Project

1. **Prepare your ZIP file**
   - Include all `.wav` audio files and a `emotion_labels.csv` file.
   - Example folder before zipping:
     ```
     violin_emotion_data/
       ├── audio1.wav
       ├── audio2.wav
       ├── audio3.wav
       └── emotion_labels.csv
     ```
   - Compress into `your-data.zip`.

2. **Upload to Kaggle**
   - In your Kaggle notebook, go to *Add Input → Upload*.
   - Upload `your-data.zip`.

3. **Unzip and Run**
   - Cell 1.5 automatically unzips the dataset and verifies structure.
   - Run all cells from top to bottom.

4. **Add New Audio for Testing**
   - Place new `.wav` files in `/kaggle/working/violin-emotion-analysis/new_audio/`.
   - Run Cell 6 to simulate emotion predictions.

5. **Retrain (Optional)**
   - Replace your dataset and CSV in `/data/`.
   - Rerun all cells from Cell 1.5 onward to train from scratch.

## Dependencies
Installed automatically in Cell 1:
librosa
scikit-learn
matplotlib
numpy
pandas
noisereduce
tensorflow
shap
joblib

yaml

---

## Outputs
After training, you’ll have:
/models/
├── lstm_hybrid_model.h5
├── rf_model.pkl
└── scaler.pkl

yaml
---

## Author
**Jazmine Gu**  
Raleigh Charter High School 2025
