from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import numpy as np
import pandas as pd
import pickle
import os
import json
from tempfile import NamedTemporaryFile

app = FastAPI(title="Body Language Analysis API")

# Try to load the model, but have a fallback
try:
    with open('body_language.pkl', 'rb') as f:
        model = pickle.load(f)
    model_loaded = True
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    model_loaded = False

@app.get("/")
async def root():
    return {
        "message": "Body Language Analysis API",
        "status": "running",
        "model_loaded": model_loaded,
        "endpoints": {
            "GET /": "This info page",
            "GET /health": "Health check endpoint",
            "POST /upload/": "Upload a video for analysis"
        }
    }

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model_loaded}

@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    if not model_loaded:
        raise HTTPException(status_code=500, detail="Model not loaded properly")
    
    # Check if the file is a video
    if not file.content_type.startswith('video/'):
        return {"error": "Uploaded file is not a video"}
    
    # Save file temporarily
    temp = NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        contents = await file.read()
        temp.write(contents)
        temp.close()
        
        # For the initial version, just return that we received the file
        # This can be expanded later with actual video processing
        file_size = os.path.getsize(temp.name)
        
        return {
            "message": "Video received successfully",
            "filename": file.filename,
            "content_type": file.content_type,
            "file_size_bytes": file_size,
            "model_status": "loaded" if model_loaded else "not loaded",
            "note": "Video processing with OpenCV and MediaPipe will be implemented in the next version"
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
