# Emotion Detection System

## Project Overview
The Emotion Detection System is a deep learning-based application that recognizes human emotions from facial expressions in real-time using a webcam.  
It uses a Convolutional Neural Network (CNN) model trained to classify seven emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

---

## Features
- Real-time emotion detection using a webcam.
- Detects faces using OpenCV Haar Cascade Classifier.
- Classifies emotions with a pre-trained CNN model.
- Displays the detected emotion and confidence percentage.

---

## Project Structure
Emotion-Detection/
│
├── detect_emotion.py # Main program to run emotion detection
├── analyze_h5.py # Script to analyze the .h5 model structure
├── deep_inspect_h5.py # Script to inspect model weights and layers
├── inspect_model.py # Script for top-level model inspection
├── model_emotion_full.h5 # Pre-trained CNN model file
├── requirements.txt # Dependencies list
└── README.md # Project documentation

yaml
Copy code

---

## Requirements
The project requires the following Python libraries:
tensorflow
opencv-python
numpy

cpp
Copy code

Install them using:
```bash
pip install -r requirements.txt
How to Run
Clone the repository:

bash
Copy code
git clone https://github.com/<your-username>/Emotion-Detection.git
cd Emotion-Detection
Ensure the model file model_emotion_full.h5 is in the same directory as the scripts.

Run the main program:

bash
Copy code
python detect_emotion.py
The webcam will open and start detecting emotions.

Press Q to exit the application.

Model Description
Input: Grayscale image of size 48x48.

Architecture:

Convolutional layers with ReLU activation.

MaxPooling and Dropout layers for regularization.

Fully connected Dense layers.

Softmax output layer for 7 emotion classes.

Emotions Classified:

Angry

Disgust

Fear

Happy

Sad

Surprise

Neutral

Usage Notes
Ensure good lighting conditions for better accuracy.

Keep your face clearly visible to the webcam.

Works best with neutral background and clear facial expressions.

Author
Jay Durga Surendra Gowda Veeri
Emotion Detection System Project – Deep Learning and Computer Vision

