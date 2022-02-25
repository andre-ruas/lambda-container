import json
import boto3
import base64
import os
import pandas as pd

s3 = boto3.resource("s3")
bucket_name = os.environ["S3_BUCKET"]
bucket = s3.Bucket(bucket_name)
lambda_client = boto3.client("lambda")

def data_missing_response():
    return {
        "statusCode": 400,
        "body": "input data missing",
    }

def decode_to_dataframe(data):
    decoded_csv = base64.b64decode(data, validate=True)
    local_path = '/tmp/data.csv'
    with open(local_path, "wb") as f:
        f.write(decoded_csv)
    return pd.read_csv(local_path, encoding="unicode_escape")

def save_data(dataFrame, model_id):
    local_data_path = "/tmp/data.csv"
    dataFrame.to_csv(local_data_path, index=False)
    with open(local_data_path, "rb") as f:
        return bucket.upload_fileobj(f, model_id + '.csv')

def handler(event, context):

    data = json.loads(event['body'])

    try:
        model_id = data["model_id"]
        dataset = data["dataset"]
    except:
        return data_missing_response()
    
    df = decode_to_dataframe(dataset)

    save_data(df,model_id)

    return {
        "statusCode": 200,
        "body": json.dumps({
            'message': 'Modelo salvo com sucesso!',
            'model': model_id
        }),        
    }
