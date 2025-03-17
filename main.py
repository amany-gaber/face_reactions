from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import cv2
import numpy as np
import pandas as pd
import pickle
import os
from tempfile import NamedTemporaryFile

app = FastAPI()

# Load the trained model
try:
    with open('body_language.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")
    # Create a placeholder model
    model = None

@app.get("/")
async def root():
    return {"message": "Body Language Analysis API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded properly")
    
    # Check if the file is a video
    if not file.content_type.startswith('video/'):
        return {"error": "Uploaded file is not a video"}
    
    # Save file temporarily
    temp = NamedTemporaryFile(delete=False, suffix=".mp4")
    temp.write(await file.read())
    temp.close()
    
    # Process video
    cap = cv2.VideoCapture(temp.name)
    
    # Just return basic video info for now
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    cap.release()
    os.remove(temp.name)
    
    return {
        "video_info": {
            "fps": fps,
            "frame_count": frame_count,
            "duration_seconds": duration
        },
        "model_status": "loaded" if model is not None else "not loaded"
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
