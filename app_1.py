#!/usr/bin/env python

from flask import Flask
from waitress import serve
from sklearn.externals import joblib
from sklearn import preprocessing
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
from flask import request
import json
import random
import decimal
from flask_cors import CORS
from flask import send_file
from flask import Response
from keras import backend as K

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/predict-churn", methods=['POST'])
def predict_churn():
    model = joblib.load("churn-model.pkl")
    ToPredictdf = load_jsonData()
    X = ToPredictdf.values.astype(np.float)
    new_prediction = model.predict(X)
    new_prediction = (new_prediction > 0.5)    
    print(new_prediction)
    K.clear_session()
    return str(new_prediction)

@app.route("/predict-churn-batch-actual", methods=['POST'])
def predict_churn_batch_actual():
    model = joblib.load("churn-model.pkl")
    ToPredictdf = load_jsonData()
    X = ToPredictdf.as_matrix().astype(np.float)
    #scaler = preprocessing.StandardScaler()
    #X = scaler.fit_transform(X)
    
    Predictiondf = pd.DataFrame(columns=['CUSTOMER.CODE','PredictedValue'])
    for i in range(len(X)):
        new_prediction = model.predict_proba(X[[i]])
        new_prediction = (new_prediction > 0.5)
        Predictiondf = Predictiondf.append({'CUSTOMER.CODE': ToPredictdf['CUSTOMER.CODE'][i],'PredictedValue':new_prediction[0][0]}, ignore_index=True)
    out = Predictiondf.to_json(orient='records')    
    response = app.response_class(response=out,status=200,mimetype='application/json')
    return response
    
@app.route("/predict-churn-batch", methods=['POST'])
def predict_churn_batch_mocked():
    model = joblib.load("churn-model.pkl")
    ToPredictdf = load_jsonData()
    X = ToPredictdf.as_matrix().astype(np.float)
    #scaler = preprocessing.StandardScaler()
    #X = scaler.fit_transform(X)
    Predictiondf = pd.DataFrame(columns=['CUSTOMER.CODE','PredictedValue'])
    for i in range(len(X)):
        #new_prediction = model.predict_proba(X[[i]])
        #new_prediction = (new_prediction > 0.5)
        new_prediction_mock = random.randint(10, 99)/100
        if (new_prediction_mock < 0.5):
            new_prediction_out= "Low Risk"
        elif (new_prediction_mock > 0.5 and new_prediction_mock < 0.8):
            new_prediction_out= "Medium Risk"
        else:
            new_prediction_out= "High Risk"
        Predictiondf = Predictiondf.append({'CUSTOMER.CODE': ToPredictdf['CUSTOMER.CODE'][i],'PredictedCategory':new_prediction_out,'PredictedValue':new_prediction_mock}, ignore_index=True)
    out = Predictiondf.to_json(orient='records')       
    response = app.response_class(response=out,status=200,mimetype='application/json')
    K.clear_session()
    return response
		
def load_jsonData():
    discreet_encoding_values = joblib.load("discreet_encoding_values.pkl")
    json_data = request.get_json()
    data = json_data["X"]
    ToPredictdf = pd.DataFrame.from_dict(json_normalize(data), orient='columns') 
    print('discreet_encoding_values :: ', discreet_encoding_values)
    print('ToPredictdf :: ', ToPredictdf)

    ToPredictdf.reset_index(level=0, inplace=True)
    for i, row in discreet_encoding_values.iterrows():
        ToPredictdf[discreet_encoding_values[0][i]] = ToPredictdf[discreet_encoding_values[0][i]].map(dict(discreet_encoding_values[1][i]))
    ToPredictdf.drop(["index"], axis = 1, inplace=True) 
    return ToPredictdf

@app.route("/customer-tweets/<int:customer_code>")
def get_customer_tweets(customer_code):
    tweets_df = joblib.load("customer-tweets.pkl")
    customer_tweets = tweets_df.loc[tweets_df['Customer.Code'] == customer_code]
    out = customer_tweets.to_json(orient='records')
    print(out)
    response = app.response_class(response=out,status=200,mimetype='application/json')
    return response
 
@app.route('/get_interactive_image/<string:filename>')
def get_interactive_image(filename):
    return send_file(filename, mimetype='text/html')
	
@app.route('/get_image/<string:filename>')
def get_image(filename):
    filename = filename + '.png'
    return send_file(filename, mimetype='image/png')
	
# This is important so that the server will run when the docker container has been started. 
# Host=0.0.0.0 needs to be provided to make the server publicly available.
if __name__ == "__main__":
    serve(app,host='0.0.0.0', port=5000)
