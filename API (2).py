from fastapi import FastAPI, File, UploadFile
import uvicorn
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import pickle
import os
from tempfile import NamedTemporaryFile

app = FastAPI()

# Load the trained model
with open('body_language.pkl', 'rb') as f:
    model = pickle.load(f)

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    # Save file temporarily
    temp = NamedTemporaryFile(delete=False, suffix=".mp4")
    temp.write(await file.read())
    temp.close()
    
    # Process video
    cap = cv2.VideoCapture(temp.name)
    
    results_list = []
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False  
            
            # Make Detections
            results = holistic.process(image)
            
            try:
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                
                face = results.face_landmarks.landmark
                face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
                
                row = pose_row + face_row
                
                X = pd.DataFrame([row])
                body_language_class = model.predict(X)[0]
                body_language_prob = model.predict_proba(X)[0]
                
                results_list.append({
                    "class": body_language_class,
                    "probability": round(body_language_prob[np.argmax(body_language_prob)], 2)
                })
            except:
                pass
    
    cap.release()
    os.remove(temp.name)
    return {"results": results_list}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
