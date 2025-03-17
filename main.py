from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import numpy as np
import pickle
import os
from tempfile import NamedTemporaryFile

app = FastAPI(title="Body Language Analysis API")

# Initialize the model as None
model = None

# Flag to track OpenCV availability
cv2_available = False

# Try to load OpenCV
try:
    import cv2
    cv2_available = True
    print("OpenCV loaded successfully")
except ImportError:
    print("OpenCV could not be loaded - running in limited mode")

# Try to load the model, but don't fail if it doesn't work
try:
    with open('body_language.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    print("Continuing without model for testing purposes")

# Functions that depend on OpenCV
if cv2_available:
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

    # A function to preprocess frames (e.g., resizing, normalization)
    def preprocess_frame(frame):
        # Resize the frame to the model's expected input size
        resized_frame = cv2.resize(frame, (224, 224))
        # Normalize pixel values to [0, 1]
        normalized_frame = resized_frame / 255.0
        # Add a batch dimension
        return np.expand_dims(normalized_frame, axis=0)

    # A function to process the video frames and make predictions using the model
    def process_video_and_predict(video_file_path):
        # Extract frames from the video
        frames = extract_frames(video_file_path)
        
        # List to store predictions
        predictions = []
        
        for frame in frames:
            processed_frame = preprocess_frame(frame)
            # Make a prediction using the loaded model
            prediction = model.predict(processed_frame)
            
            # Get the predicted class and highest probability
            predicted_class = np.argmax(prediction, axis=1)
            predicted_prob = np.max(prediction)
            
            predictions.append({
                "class": str(predicted_class[0]),
                "probability": float(predicted_prob)
            })
        
        return predictions

@app.get("/")
async def root():
    return {
        "message": "Body Language Analysis API",
        "status": "running",
        "model_loaded": model is not None,
        "opencv_available": cv2_available,
        "test_mode": model is None or not cv2_available,
        "endpoints": {
            "GET /": "This info page",
            "GET /health": "Health check endpoint",
            "POST /upload/": "Upload a video for analysis"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "ok", 
        "model_loaded": model is not None,
        "opencv_available": cv2_available
    }

@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    # Check if the file is a video
    if not file.content_type.startswith('video/'):
        return {"error": "Uploaded file is not a video"}
    
    # Check if OpenCV is available
    if not cv2_available:
        return {"error": "OpenCV is not available, video analysis is disabled"}
    
    # Check if model is loaded
    if model is None:
        return {"error": "Model not loaded"}
    
    # Save file temporarily
    temp = NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        contents = await file.read()
        temp.write(contents)
        temp.close()
        
        file_size = os.path.getsize(temp.name)
        
        # Process the video and get predictions using the model
        predictions = process_video_and_predict(temp.name)
        
        return {
            "message": "Video received successfully",
            "filename": file.filename,
            "content_type": file.content_type,
            "file_size_bytes": file_size,
            "mode": "production",
            "results": predictions
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
