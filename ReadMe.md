# Deepfake Video Analyzer

This project provides a Streamlit-based interface for analyzing videos to detect deepfake content. It uses a pre-trained deepfake detection model to make predictions on uploaded videos.

## Summary

The Deepfake Analyzer project aims to detect deepfake videos by leveraging a combination of Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs). The project utilizes a dataset sourced from Kaggle containing both real and fake video samples. The dataset is preprocessed, and the video shots (images) are displayed within the Google Colab notebook for demonstration purposes.

### Dataset

The dataset used in this project is sourced from Kaggle and consists of real and fake video samples. Each video is divided into shots, and each shot is represented by a sequence of frames (images). The dataset is preprocessed to extract the video shots, which are used for model training and testing.

### Model Architecture

The deepfake detection model is based on a CNN-RNN architecture. The CNN component extracts features from individual frames of the video shots, while the RNN component analyzes the temporal sequence of these features to make predictions. The model is trained on the extracted video shot data and tested to evaluate its performance in distinguishing between real and fake videos.

### Implementation Steps

1. **Data Preparation:** Kaggle dataset is downloaded and preprocessed to extract video shots. Video shots (images) from both real and fake videos are loaded into the notebook.
2. **Data Visualization:** Demonstration of video shots by playing sequences of frames in the notebook. Visualization of both real and fake video shots to understand the characteristics of each.
3. **Model Building:** Construction of the CNN-RNN based deepfake detection model. Compilation of the model with appropriate loss function and optimization algorithm.
4. **Model Training:** Training of the model using the preprocessed dataset. Monitoring of training progress with metrics such as loss and accuracy.
5. **Model Evaluation:** Testing of the trained model on a separate test dataset. Evaluation of model performance using metrics such as accuracy, precision, recall, and F1-score.

### Conclusion

The Deepfake Analyzer project demonstrates the application of deep learning techniques for detecting deepfake videos. By leveraging a CNN-RNN based model trained on a Kaggle dataset, the project aims to accurately differentiate between real and fake videos. The provided documentation outlines the project's objectives, dataset, model architecture, and implementation steps, facilitating a clear understanding of the project's methodology and outcomes.

## Folder Structure

- `deepfake_main_run.py`: Main file for the Streamlit interface. This file contains the code to run the Streamlit app. To run the application, execute the following command:

    ```
    streamlit run deepfake_main_run.py
    ```

- `dfvenv`: Virtual environment for the project. To activate this environment, use the following command:

    ```
    source dfvenv/bin/activate
    ```

- `interface_deepfake.py`: Helper file containing functions and utilities for the Streamlit interface.

- `my_model.keras`: Trained deepfake detection model saved in Keras format.

- `requirements.txt`: File containing the dependencies required to run the project. To install the dependencies, use the following command:

    ```
    pip install -r requirements.txt
    ```

## Usage

1. Activate the virtual environment:

    ```
    source dfvenv/bin/activate
    ```

2. Install the required dependencies:

    ```
    pip install -r requirements.txt
    ```

3. Run the Streamlit interface:

    ```
    streamlit run deepfake_main_run.py
    ```

4. Upload a video to analyze for deepfake content.

## Author

Aniket
