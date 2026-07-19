"""
Gradio frontend for the Team Tier Predictor.
Collects all 11 input fields, sends them to the FastAPI backend
running at http://localhost:8000/predict, and shows the result.
"""

import requests
import gradio as gr

API_URL = "http://localhost:8000/predict"


def predict_tier(squad_avg_rating, squad_min_rating, squad_rating_std, rating_range,
                  buildUpPlayPassing, defencePressure, chanceCreationShooting,
                  played, wins, losses, draws):

    payload = {
        "squad_avg_rating": squad_avg_rating,
        "squad_min_rating": squad_min_rating,
        "squad_rating_std": squad_rating_std,
        "rating_range": rating_range,
        "buildUpPlayPassing": buildUpPlayPassing,
        "defencePressure": defencePressure,
        "chanceCreationShooting": chanceCreationShooting,
        "played": played,
        "wins": wins,
        "losses": losses,
        "draws": draws,
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=5)
        response.raise_for_status()
        result = response.json()
        tier = result["predicted_tier"]
        probs = result["probabilities"]
        probs_text = "\n".join([f"{k}: {v:.1%}" for k, v in probs.items()])
        return f"Predicted Tier: {tier}\n\n{probs_text}"
    except Exception as e:
        return f"Error calling API: {e}"


demo = gr.Interface(
    fn=predict_tier,
    inputs=[
        gr.Number(label="Squad Avg Rating", value=68),
        gr.Number(label="Squad Min Rating", value=58),
        gr.Number(label="Squad Rating Std", value=4),
        gr.Number(label="Rating Range", value=10),
        gr.Number(label="Build Up Play Passing", value=50),
        gr.Number(label="Defence Pressure", value=50),
        gr.Number(label="Chance Creation Shooting", value=50),
        gr.Number(label="Played", value=38, precision=0),
        gr.Number(label="Wins", value=15, precision=0),
        gr.Number(label="Losses", value=15, precision=0),
        gr.Number(label="Draws", value=8, precision=0),
    ],
    outputs=gr.Textbox(label="Result"),
    title="Team Tier Predictor",
    description="Enter squad, tactical, and season record stats to predict the team's tier (Top / Mid / Bottom).",
)

if __name__ == "__main__":
    demo.launch()