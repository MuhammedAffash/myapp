from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()

# Load the trained model
model = joblib.load("model/diabetes_detection_model.joblib")
scaler = joblib.load("model/scaler.joblib")
# Define input data model with default values
class InputData(BaseModel):
    gender: int
    age: float
    hypertension: int
    heart_disease: int
    bmi: float
    HbA1c_level: float
    blood_glucose_level: float
    No_Info: int = 0
    current: int = 0
    ever: int = 0
    former: int = 0
    never: int = 0
    not_current: int = 0

@app.post("/predict/")
def predict_diabetes(data: InputData):
    numerical_features = np.array([[
    data.age , data.bmi, data.HbA1c_level, data.blood_glucose_level
    ]])
    categorical_features = np.array([[data.gender,data.hypertension, data.heart_disease, data.No_Info, data.current, data.ever, data.former, data.never,
       data.not_current]])

    
    numerical_features_scaled = scaler.transform(numerical_features)
    features_scaled = np.concatenate([numerical_features_scaled, categorical_features], axis=1)    
    prediction_probabilities = model.predict_proba(features_scaled).flatten()
    index = np.argmax(prediction_probabilities)
    confidence = round(prediction_probabilities[0], 4) * 100
    
    
    return {"prediction": int(index), "confidence": confidence}
