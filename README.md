# YouTube Media Control using Gesture Classification

Gesture Classification trained with LSTM (Long Short Term Memory)
Using Mediaipipe Handpose Detection
+ PyAutoGUI

## Flow
Hands gesture --> camera --> Mediapipe hands --> LSTM --> Predict Gesture --> PyAutoGUI --> Youtube

## Dataset
collected from "https://www.kaggle.com/datasets/imsparsh/gesture-recognition"
using 3 gesture only (Stop, Thumbs Down, Thumbs Up)

## Preprocessing Dataset
There is several process in "Training Dataset.ipynb"
1. Download dataset from kaggle
2. Pick Gesture class we want to use 
3. Extract mediapipe feature from each class into X_train.npy, y_train.npy
4. Define LSTM architecture
5. Adjust parameter to train
6. Download the trained h5 files

## How to run the project
1. Install requirements
2. Open Main.py
3. choose the h5 model, adjust it in the line code
3. Setup the camera
4. Run
5. Open YouTube
6. Do 3 gesture in front of camera, open palms for stop/play, thumbs up to volume up, thumbs down for volume down. (left hands)
7. press q to exit programs