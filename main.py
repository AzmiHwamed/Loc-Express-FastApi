from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from google import genai
from pydantic import BaseModel
from typing import List, Union

app = FastAPI()
class PropertyLocation(BaseModel):
    governorate: str
    city: str
    Locality: str
    Street: str

class PropertyFeature(BaseModel):
    name: str
    value: Union[str, int, bool]

class PropertyDetails(BaseModel):
    title: str
    description: str
    propertyType: str
    price: int
    surface: int
    propertyLocation: PropertyLocation
    propertyFeatures: List[PropertyFeature]
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


@app.post("/detect")
async def detect_spam(details: PropertyDetails):
    try:
        client = genai.Client(api_key="AIzaSyCTmf2trLBuQqqLwMacvI3hJ0AHUj6zkdc")
        print(details.model_dump_json())
        prompt = f"""
        Given the following property details, determine if this is likely a legitimate real estate listing or spam. 
        Pay attention to the title and description it may be tricky, but don't overthink the realtion or missmatch between the title and description , just please see if the title and decription phrases contextes are only related to real estate. don't take into consediration the relation between the title and description, just see if they are related to real estate or not.
        just return true or false.
        Object:
        {details.model_dump_json()}
        """

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return {"is_legit": response.text.strip().lower() == "true"}
    except Exception as e:
        return {"error": str(e)}
        
