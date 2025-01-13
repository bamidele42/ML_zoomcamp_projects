from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
dv = DictVectorizer(sparse=False)

# Initialize FastAPI

app = FastAPI()

# Define the request body format for prediction


class PredictionFeatures(BaseModel):
    limit_bal: float
    sex: str
    education: str
    marriage: str
    age: float
    pay_0: float
    pay_2: float
    pay_3: float
    pay_4: float
    pay_5: float
    pay_6: float
    bill_amt1: float
    bill_amt2: float
    bill_amt3: float
    bill_amt4: float
    bill_amt5: float
    bill_amt6: float
    pay_amt1: float
    pay_amt2: float
    pay_amt3: float
    pay_amt4: float
    pay_amt5: float
    pay_amt6: float

# Global variable to store the loaded model
# model = None

# Download the model
# def download_model():
#     global model
#     with open('app/model.bin', 'rb') as f_in:
#         dv, rf = pickle.load(f_in)

# Download the model immediately when the script runs
# download_model()


with open('app/model.bin', 'rb') as f_in:
    dv, rf = pickle.load(f_in)

# API Root endpoint


@app.get("/")
async def index():
    return {"message": " Welcome to the Customer Payment Default API. Use the /predict"}

# Prediction endpoint


@app.post("/predict")
async def predict(features: PredictionFeatures):
    # create input DataFrame for prediction
    input_data = pd.DataFrame([{
        "limit_bal": features.limit_bal,
        "sex": features.sex,
        "education": features.education,
        "marriage": features.marriage,
        "age": features.age,
        "pay_0": features.pay_0,
        "pay_2": features.pay_2,
        "pay_3": features.pay_3,
        "pay_4": features.pay_4,
        "pay_5": features.pay_5,
        "pay_6": features.pay_6,
        "bill_amt1": features.bill_amt1,
        "bill_amt2": features.bill_amt2,
        "bill_amt3": features.bill_amt3,
        "bill_amt4": features.bill_amt4,
        "bill_amt5": features.bill_amt5,
        "bill_amt6": features.bill_amt6,
        "pay_amt1": features.pay_amt1,
        "pay_amt2": features.pay_amt2,
        "pay_amt3": features.pay_amt3,
        "pay_amt4": features.pay_amt4,
        "pay_amt5": features.pay_amt5,
        "pay_amt6": features.pay_amt6
    }])

    # Prediction using the loaded model
    customer = input_data.to_dict()
    X = dv.transform([customer])
    y_pred = rf.predict_proba(X)[0, 1]

    if y_pred < 0.5:
        return "This customer is not likely to default Payment"
    else:
        return "This customer is likely to default Payment"


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
