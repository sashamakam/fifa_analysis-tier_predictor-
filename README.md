# FIFA Team Tier Predictor

An exploratory machine-learning project that classifies team tiers using FIFA squad ratings, tactics, and season results. A FastAPI endpoint returns the predicted tier and class probabilities. A Gradio interface provides an interactive demo.

## Model comparison

The [training notebook](model_training.ipynb) uses a stratified 80/20 split with 287 held-out examples and a random seed of 42.

| Model | Test accuracy |
| --- | ---: |
| Random forest | 90.59% |
| Logistic regression | 88.85% |
| Multilayer perceptron | 84.67% |
| Decision tree | 81.53% |

The notebook refits the selected random forest on all rows before exporting it. The reported evaluation uses season results as inputs, including wins, draws, and losses.

## Run locally

From the repository root, create and activate a Python virtual environment, then run:

```sh
python -m pip install fastapi uvicorn joblib pandas numpy scikit-learn gradio requests
python -m uvicorn backend:app --port 8000
```

In a second terminal with the environment active, run `python frontend.py` and open the Gradio URL it prints. The API docs are at <http://localhost:8000/docs>.

The notebook reads [cleandata.csv](cleandata.csv). To rerun it, also install `jupyter matplotlib seaborn`. Use the same scikit-learn version as the saved artifacts or regenerate them if a version mismatch prevents loading.
