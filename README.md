# Sign2Speech: Real-Time Sign Language to Speech Converter

## Project Overview
**Sign2Speech** is a real-time system that converts sign language gestures into spoken words. It is designed to help hearing-impaired individuals communicate more easily by providing both **audio** and **visual feedback** for gestures. This project combines **machine learning, computer vision, and text-to-speech** technologies to create an inclusive communication tool.

## Features
- Real-time **gesture recognition** using Mediapipe hand tracking
- **KNN machine learning model** trained on 20 common sign language gestures
- Converts gestures into **speech** using pyttsx3 Text-to-Speech
- Displays **visual feedback** of the recognized gesture alongside the webcam feed
- Runs on a standard computer with a webcam

## Technologies Used
- Python
- Mediapipe (hand tracking)
- OpenCV (webcam and image handling)
- scikit-learn (KNN gesture classification)
- pyttsx3 (Text-to-Speech)

## How It Works
1. The system detects hand landmarks from the webcam feed using Mediapipe.
2. The KNN model classifies the gesture from the detected landmarks.
3. Once recognized, the system converts the gesture into spoken words using pyttsx3.
4. Simultaneously, the corresponding gesture is displayed on screen for visual feedback.

## Applications
- Assistive tool for hearing-impaired individuals
- Educational tool to teach sign language
- Public service for faster, intuitive communication

## Installation
1. Clone the repository:
   ```bash
      git clone https://github.com/bsujithasree-gif/Sign2Speech.git
[Watch Sign2Speech Demo](https://github.com/bsujithasree-gif/Sign2Speech/blob/main/sign2speech.mp4?raw=true)



  


