# Facial Recognition System – IE4428 Assignment 2

## Features
- Web interface with live camera preview  
- Guided face image capturing with silhouette overlay  
- Real-time face recognition using ArcFace & DeepFace  
- Automatic matching with custom face database  
- Lightweight and easy to deploy

## Tech Stack

- **Frontend**: HTML5 + CSS + Vanilla JS  
- **Backend**: Flask (Python)  
- **Face Recognition**: DeepFace (`ArcFace`, `opencv` backend)  
- **Video Processing**: OpenCV  
- **ML**: Cosine similarity on embeddings

## Getting Started
git clone git@github.com:sienzzzz/FacialRecognition.git
cd FacialRecognition

# Create Virtual Environment
python3 -m venv projvenv
source projvenv/bin/activate
pip install -r requirements.txt

## Run the app
python app.py
