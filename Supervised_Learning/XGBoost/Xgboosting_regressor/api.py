from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np

app = FastAPI(
    title="House Price Prediction API",
    description="REST API for predicting house prices using an XGBoost regression model",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("xgboost_house_price_model.pkl")

class HouseInput(BaseModel):
    GrLivArea: float = Field(..., gt=0, lt=10000)
    BedroomAbvGr: int = Field(..., ge=0, le=20)
    FullBath: int = Field(..., ge=0, le=10)
    TotalBsmtSF: float = Field(..., ge=0, lt=10000)
    GarageCars: int = Field(..., ge=0, le=10)
    YearBuilt: int = Field(..., ge=1800, le=2025)
    LotArea: float = Field(..., gt=0, le=100000)
    OverallQual: int = Field(..., ge=1, le=10)

class PredictionResponse(BaseModel):
    prediction: float
    formatted_prediction: str

@app.get("/")
def root():
    return {"status": "ok", "service": "house-price-predictor"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
def predict_price(input_data: HouseInput):
    try:
        X = np.array([[ 
            input_data.GrLivArea,
            input_data.BedroomAbvGr,
            input_data.FullBath,
            input_data.TotalBsmtSF,
            input_data.GarageCars,
            input_data.YearBuilt,
            input_data.LotArea,
            input_data.OverallQual
        ]])

        pred = model.predict(X)[0]

        return PredictionResponse(
            prediction=round(float(pred), 2),
            formatted_prediction=f"${pred:,.2f}"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))