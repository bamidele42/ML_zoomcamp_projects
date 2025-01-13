from fastapi import FastAPI
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
# dv = DictVectorizer(sparse=False)

with open("model.bin", "rb") as f_in:
    dv, rf = pickle.load(f_in)

app = FastAPI()

@app.get('/')
def reed_root():
    return {'message': 'Customer default Payment model API'}

@app.post('/predict')
def predict(data):
    """
    Predicts maybe a customer will default on hiis Payment
    """
    # customer = data.to_dict()
    X = dv.transform([customer])
    y_pred = rf.predict_proba(X)[0, 1]

    if y_pred < 0.5:
        return "This customer is not likely to default Payment"
    else:
        return "This customer is likely to default Payment"

