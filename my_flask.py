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
from flask import send_file
import random
app = Flask(__name__)


@app.route("/customer-tweets/<int:customer_code>")
def get_customer_tweets(customer_code):
    tweets_df = joblib.load("customer-tweets.pkl")
    customer_tweets = tweets_df.loc[tweets_df['Customer.Code'] == customer_code]
    out = customer_tweets.to_json(orient='records')
    print(out)
    response = app.response_class(response=out,status=200,mimetype='application/json')
    return response

	
@app.route('/get_image/<string:filename>')
def get_image(filename):
    filename = filename + '.png'
    return send_file(filename, mimetype='image/png')
	
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
	
@app.route("/predict-churn-batch", methods=['POST'])
def predict_churn_batch_mocked():
    ToPredictdf = load_jsonData()
    X = ToPredictdf.as_matrix().astype(np.float)
    Predictiondf = pd.DataFrame(columns=['CUSTOMER.CODE','PredictedValue'])
    for i in range(len(X)):
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
    return response	


# This is important so that the server will run when the docker container has been started. 
# Host=0.0.0.0 needs to be provided to make the server publicly available.
if __name__ == "__main__":
    serve(app,host='0.0.0.0', port=5000)
