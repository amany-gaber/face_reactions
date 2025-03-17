from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import numpy as np
import pandas as pd
import pickle
import os
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
        
        # Add test results if model is not available
        test_results = []
        if model is None:
            test_results = [
                {"class": "happy", "probability": 0.85},
                {"class": "neutral", "probability": 0.12},
                {"class": "sad", "probability": 0.03}
            ]
        
        return {
            "message": "Video received successfully",
            "filename": file.filename,
            "content_type": file.content_type,
            "file_size_bytes": file_size,
            "mode": "test" if model is None else "production",
            "results": test_results if model is None else "Would process with real model"
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
