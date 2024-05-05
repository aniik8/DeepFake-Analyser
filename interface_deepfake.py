import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import keras

IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 10

MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048

def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]


def build_feature_extractor():
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


feature_extractor = build_feature_extractor()
def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)

def prepare_single_video(frames):
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
# Function to preprocess video frames
# def preprocess_frame(frame):
#     # Add your preprocessing steps here (e.g., resizing, normalization)
#     # Example: resized_frame = cv2.resize(frame, (new_width, new_height))
#     return frame


def sequence_prediction(path):
    frames = load_video(os.path.join(DATA_FOLDER, TEST_FOLDER,path))
    frame_features, frame_mask = prepare_single_video(frames)
    return model.predict([frame_features, frame_mask])[0]

# Function to make prediction using the loaded model
def predict_deepfake(frame):
    # Assuming your model takes preprocessed frames as input
    # Example: prediction = model.predict(np.expand_dims(frame, axis=0))
    # You might need to reshape the prediction depending on your model output
    prediction = False
    if(sequence_prediction(test_video)<=0.5):
        prediction = True
    return prediction

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
        video_array = np.frombuffer(video_bytes, dtype=np.uint8)
        video = cv2.imdecode(video_array, cv2.IMREAD_UNCHANGED)

        # Display the uploaded video
        st.video(video_bytes)

        # Button to process the video
        if st.button("Analyze Video"):
            # Iterate through video frames
            for frame in video:
                # Preprocess each frame
                preprocessed_frame = preprocess_frame(frame)

                # Make prediction using the model
                prediction = predict_deepfake(preprocessed_frame)

                # Display the prediction
                st.write(prediction)

if __name__ == "__main__":
    main()
