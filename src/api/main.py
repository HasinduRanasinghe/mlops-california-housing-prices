from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.train import HousingPriceModel
from src.data.preprocessing import DataPreprocessor
from src.models.train import train_and_save_model
from sklearn.preprocessing import StandardScaler


class HousingPredictionRequest(BaseModel):
    features: list[float]


class HousingPredictionResponse(BaseModel):
    predicted_price: float


app = FastAPI()


def ensure_model_exists():
    model_path = 'models/housing_price_model.pth'
    if not os.path.exists(model_path):
        print("Model not found. Training model...")
        train_and_save_model()


@app.on_event("startup")
def startup_event():
    ensure_model_exists()

    global model, scaler

    # Load model
    model = HousingPriceModel(input_dim=8)
    model.load_state_dict(torch.load('models/housing_price_model.pth'))
    model.eval()

    # Recreate scaler
    preprocessor = DataPreprocessor()
    _, X_test_scaled, _, _ = preprocessor.preprocess()
    scaler = StandardScaler().fit(X_test_scaled)


@app.post("/predict")
def predict_housing_price(request: HousingPredictionRequest):
    try:
        # Validate input
        if len(request.features) != 8:
            raise HTTPException(status_code=400, detail="Invalid number of features")

        # Scale input
        input_scaled = scaler.transform([request.features])
        input_tensor = torch.FloatTensor(input_scaled)

        # Predict
        with torch.no_grad():
            prediction = model(input_tensor).numpy()[0][0]

        return HousingPredictionResponse(predicted_price=float(prediction))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "healthy"}