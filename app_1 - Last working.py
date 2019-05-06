#!/usr/bin/env python

from flask import Flask
from waitress import serve
from sklearn.externals import joblib
from sklearn import preprocessing
import numpy as np
#from Customer_Churn_Modelling_ProdSetup import model_creation

# New imports that can handle requests and json content
from flask import request
import json


app = Flask(__name__)

@app.route("/predict-churn", methods=['POST'])
def predict_iris():
    json = request.get_json()
    X = json["X"]
    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X)
    model = joblib.load("churn-model.pkl")
    new_prediction = model.predict(X)
    new_prediction = (new_prediction > 0.5)
    return str(new_prediction)
    #return str(model.predict(scaler.transform(np.array([X]))))
	
# def main():
#     return model_creation()

# This is important so that the server will run when the docker container has been started. 
# Host=0.0.0.0 needs to be provided to make the server publicly available.
if __name__ == "__main__":
    serve(app,host='0.0.0.0', port=5000)
#    main()