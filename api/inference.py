import json
import base64
import pandas as pd
import pickle
import boto3
import os
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression

s3_client = boto3.client("s3")
bucket_name = os.environ["S3_BUCKET"]


def data_missing_response():
    return {
        "statusCode": 400,
        "body": "input data missing"
    }

def decode_to_dataframe(data):
    decoded_csv = base64.b64decode(data, validate=True)
    local_path = '/tmp/data.csv'
    with open(local_path, "wb") as f:
        f.write(decoded_csv)
    return pd.read_csv(local_path, encoding="unicode_escape")

def get_model_from_s3(model_id):
    local_model_path = "/tmp/model.pkl"
    s3_client.download_file(bucket_name, model_id + ".pkl", local_model_path)
    with open(local_model_path, "rb") as file:
        return pickle.load(file)

def get_accuracy(model_id):
    local_path = "/tmp/accuracy.json"
    s3_client.download_file(bucket_name, model_id + ".json", local_path)
    with open(local_path, "r") as file:
        return json.load(file)

def handler(event, context):

    data = json.loads(event['body'])

    try:
        model_id = data["model_id"]
        input_data = data["inference"]
    except:
        return data_missing_response()

    inference_data = decode_to_dataframe(input_data)

    model = get_model_from_s3(model_id)

    inference = model.predict(inference_data.squeeze()).tolist()
    
    accuracy  = str(get_accuracy(model_id)['accuracy'])

    return {
        "statusCode": 200,
        "body": json.dumps({
            "model_accuracy": accuracy,
            "predictions": inference
        }),
    }

   
