from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React runs here by default with Vite
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load('fifa_analysis_model.pkl')
model_columns = joblib.load('model_columns.pkl')
label_encoder = joblib.load("label_encoder.pkl")


class inputdetails(BaseModel):
    squad_avg_rating: float
    squad_min_rating: float
    squad_rating_std: float
    rating_range: float
    buildUpPlayPassing: float
    defencePressure: float
    chanceCreationShooting: float
    played: int
    wins: int
    losses: int
    draws: int

@app.get("/")
def root():
    return {"status": "ok", "message": "Team Tier Predictor API is running"}


@app.post("/predict")
def predict(team: inputdetails):
    row = {
        "squad_avg_rating": team.squad_avg_rating,
        "squad_min_rating": team.squad_min_rating,
        "squad_rating_std": team.squad_rating_std,
        "rating_range": team.rating_range,
        "buildUpPlayPassing": team.buildUpPlayPassing,
        "defencePressure": team.defencePressure,
        "chanceCreationShooting": team.chanceCreationShooting,
        "played": team.played,
        "wins": team.wins,
        "losses": team.losses,
        "draws": team.draws,
    }

    X = pd.DataFrame([row])[model_columns]

    pred_encoded = model.predict(X)[0]
    tier = label_encoder.inverse_transform([pred_encoded])[0]

    probs = model.predict_proba(X)[0]
    prob_dict = {cls: round(float(p), 4) for cls, p in zip(label_encoder.classes_, probs)}

    return {"predicted_tier": tier, "probabilities": prob_dict}