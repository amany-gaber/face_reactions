from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import numpy as np
import pandas as pd
import pickle
import os
import cv2  # OpenCV for video processing
from tempfile import NamedTemporaryFile

app = FastAPI(title="Body Language Analysis API")

# Set model_loaded to True even if loading fails
model_loaded = True
model = None

# Try to load the model, but don't fail if it doesn't work
try:
    with open('body_language.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Continuing without model for testing purposes")

# A function to extract frames from the video file using OpenCV
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

# A function to process the video frames and make predictions using the model
def process_video_and_predict(video_file_path):
    # Extract frames from the video
    frames = extract_frames(video_file_path)
    
    # Assuming the model expects a certain input format. Example:
    # Resize frames, convert to grayscale, and reshape them to match the model's input requirements.
    processed_frames = []
    for frame in frames:
        resized_frame = cv2.resize(frame, (224, 224))  # Resize to the input size the model expects (example)
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        processed_frames.append(gray_frame)
    
    # Convert frames to a NumPy array (or any format expected by your model)
    input_data = np.array(processed_frames)
    
    # Use the model to predict the class of each frame (modify this part based on your model's structure)
    predictions = []
    for frame in input_data:
        prediction = model.predict(np.expand_dims(frame, axis=0))  # Example prediction
        predicted_class = np.argmax(prediction)  # Assuming the model outputs class probabilities
        predicted_prob = np.max(prediction)  # Get the highest probability
        predictions.append({
            "class": predicted_class,
            "probability": predicted_prob
        })
    
    return predictions

@app.get("/")
async def root():
    return {
        "message": "Body Language Analysis API",
        "status": "running",
        "model_loaded": model is not None,
        "test_mode": model is None,
        "endpoints": {
            "GET /": "This info page",
            "GET /health": "Health check endpoint",
            "POST /upload/": "Upload a video for analysis (test mode)"
        }
    }

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None, "test_mode": model is None}

@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    # Check if the file is a video
    if not file.content_type.startswith('video/'):
        return {"error": "Uploaded file is not a video"}
    
    # Save file temporarily
    temp = NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        contents = await file.read()
        temp.write(contents)
        temp.close()
        
        file_size = os.path.getsize(temp.name)
        
        # If model is available, process the video and predict the output
        if model is None:
            test_results = [
                {"class": "happy", "probability": 0.85},
                {"class": "neutral", "probability": 0.12},
                {"class": "sad", "probability": 0.03}
            ]
        else:
            # If the model is available, process the video and predict the output
            test_results = process_video_and_predict(temp.name)
        
        return {
            "message": "Video received successfully",
            "filename": file.filename,
            "content_type": file.content_type,
            "file_size_bytes": file_size,
            "mode": "test" if model is None else "production",
            "results": test_results
        }
    except Exception as e:
        return {"error": f"Failed to process video: {str(e)}"}
    finally:
        # Clean up the temp file
        if os.path.exists(temp.name):
            os.remove(temp.name)

# Configure the app to use the PORT environment variable set by Railway
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
