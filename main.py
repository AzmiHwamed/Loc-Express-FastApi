from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Define the input data model
class PredictionInput(BaseModel):
    offer_demand: str  # Example: 'Rental'
    nature: str  # Example: '2'
    locality: str  # Example: 'Gabes Sud'
    delegation: str  # Example: 'Gabes Sud'
    governorate: str  # Example: 'Gabes Sud'
    surface: float  # Example: 70.0

# Load the models once at startup
model = joblib.load("./random_forest_price_prediction_model.pkl")
encoder = joblib.load("./ordinal_encoder.pkl")

@app.post("/predict")
async def predict_price(data: PredictionInput):
    # Convert input data to numpy array
    input_data = np.array([[data.offer_demand, data.nature, data.locality, data.delegation, data.governorate, data.surface]])

    # Select categorical features for encoding
    input_data_to_encode = input_data[:, [0, 2, 3, 4]]
    encoded_data = encoder.transform(input_data_to_encode)

    # Replace categorical features with encoded values
    input_data[:, [0, 2, 3, 4]] = encoded_data

    # Convert to float (as model expects numeric values)
    input_data = input_data.astype(float)

    # Make prediction
    prediction = model.predict(input_data)

    return {"predicted_price": prediction[0]}
