"""
FastAPI server for handling predictive modeling requests.
"""

import asyncio
from fastapi import FastAPI
from farnsworth.analysis.predictive_modeling import load_historical_data, preprocess_data, train_model, predict_future, integrate_predictions

app = FastAPI()

@app.get("/predict")
async def predict_endpoint():
    historical_data = await load_historical_data()
    if historical_data is not None:
        processed_data = await preprocess_data(historical_data)
        if processed_data is not None:
            model = await train_model(processed_data)
            predictions = await predict_future(processed_data, model)
            await integrate_predictions(predictions)
            return {"predictions": predictions}
    return {"error": "Prediction process failed"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)