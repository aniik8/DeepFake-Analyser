"""
Deepfake Video Analyzer

Author: Aniket Sharma

This Streamlit application allows users to upload a video and analyze if it contains deepfake content.

The application preprocesses the uploaded video frames and uses a pre-trained deepfake detection model to make predictions.

"""

import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import keras
import ssl

# Disable SSL verification for downloading model weights
ssl._create_default_https_context = ssl._create_unverified_context

# Constants
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 10

MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048

def build_feature_extractor():
    """
    Builds the feature extractor model using InceptionV3.
    
    Returns:
        keras.Model: Feature extractor model.
    """
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")

def crop_center_square(frame):
    """
    Crops the center square of a frame.
    
    Args:
        frame (numpy.ndarray): Input frame.
    
    Returns:
        numpy.ndarray: Cropped frame.
    """
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    """
    Loads video frames from the given path.
    
    Args:
        path (str): Path to the video file.
        max_frames (int): Maximum number of frames to load (0 means all frames).
        resize (tuple): Target size for resizing frames.
    
    Returns:
        numpy.ndarray: Array of loaded frames.
    """
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]  # Convert BGR to RGB
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)

def prepare_single_video(frames):
    """
    Prepares a single video for prediction.
    
    Args:
        frames (numpy.ndarray): Array of video frames.
    
    Returns:
        numpy.ndarray: Frame features.
        numpy.ndarray: Frame mask.
    """
    feature_extractor = build_feature_extractor()
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_features, frame_mask

def sequence_prediction(path, model):
    """
    Makes predictions on a video using the provided model.
    
    Args:
        path (str): Path to the video file.
        model (keras.Model): Pre-trained deepfake detection model.
    
    Returns:
        numpy.ndarray: Predictions.
    """
    frames = load_video(os.path.join(path))
    frame_features, frame_mask = prepare_single_video(frames)
    return model.predict([frame_features, frame_mask])[0]

def main():
    # Load your pre-trained model
    model_path = "my_model.keras"
    model = load_model(model_path)

    # Title and description
    st.title("Deepfake Video Analyzer")
    st.write("Upload a video to analyze if it contains deepfake content.")

    # File uploader for video
    video_file = st.file_uploader("Upload a video", type=["mp4"])

    if video_file is not None:
        # Read the uploaded video
        video_bytes = video_file.read()

        # Display the uploaded video
        st.video(video_bytes)

        # Button to process the video
        if st.button("Analyze Video"):
            # Save the uploaded video to a temporary file
            with open("temp_video.mp4", "wb") as f:
                f.write(video_bytes)

            # Make prediction using the model
            prediction = sequence_prediction("temp_video.mp4", model)

            # Display the prediction
            st.write(prediction)

            # Remove the temporary video file
            os.remove("temp_video.mp4")

if __name__ == "__main__":
    main()
