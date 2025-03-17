from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import numpy as np
import pickle
import os
import cv2  # OpenCV for video processing
from tempfile import NamedTemporaryFile

app = FastAPI(title="Body Language Analysis API")

# Initialize the model as None
model = None

# Try to load the model, but don't fail if it doesn't work
try:
    with open('body_language.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")  # This will print out the actual error in the server logs
    model = None  # Make sure to set model to None if loading fails
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

# A function to preprocess frames (e.g., resizing, normalization)
def preprocess_frame(frame):
    # Resize the frame to the model's expected input size (example: 224x224)
    resized_frame = cv2.resize(frame, (224, 224))  # Adjust based on your model's input size
    # Normalize if necessary (e.g., dividing by 255 for image data)
    normalized_frame = resized_frame / 255.0  # Normalize pixel values to [0, 1]
    # Convert to the format your model expects (e.g., adding a batch dimension)
    return np.expand_dims(normalized_frame, axis=0)

# A function to process the video frames and make predictions using the model
def process_video_and_predict(video_file_path):
    # Extract frames from the video
    frames = extract_frames(video_file_path)
    
    # List to store predictions
    predictions = []
    
    for frame in frames:
        processed_frame = preprocess_frame(frame)
        # Make a prediction using the loaded model (assuming it's a classifier)
        prediction = model.predict(processed_frame)  # Adjust based on your model's method
        
        # Example: If the model outputs a probability distribution, take the max probability
        predicted_class = np.argmax(prediction, axis=1)  # Get the predicted class
        predicted_prob = np.max(prediction)  # Get the highest probability
        
        predictions.append({
            "class": str(predicted_class[0]),  # Convert the class to a string
            "probability": float(predicted_prob)  # Convert probability to float
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
            "POST /upload/": "Upload a video for analysis"
        }
    }

@app.get("/health")
async def health():
    # Add more specific messages to indicate model status
    if model is None:
        return {"status": "ok", "model_loaded": False, "error": "Model loading failed"}
    return {"status": "ok", "model_loaded": True}

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
        
        # If the model is available, process the video and predict the output
        if model is None:
            return {"error": "Model not loaded"}
        
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
