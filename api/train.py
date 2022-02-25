import json
import boto3
import pickle
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

s3 = boto3.resource("s3")
s3_client = boto3.client("s3")
bucket_name = os.environ["S3_BUCKET"]
bucket = s3.Bucket(bucket_name)

def save_model(model, model_id):
    local_path = "/tmp/model.pkl"
    with open(local_path, "wb") as file:
        pickle.dump(model, file)
    with open(local_path, "rb") as f:
        return bucket.upload_fileobj(f, model_id + '.pkl')

def get_DataFrame_s3(file_name):
    local_data_path = "/tmp/data.csv"
    s3_client.download_file(bucket_name, file_name, local_data_path)
    return pd.read_csv(local_data_path)

def train_model(df):

    feature = df.columns[-2]
    predict = df.columns[-1]

    text_clf = Pipeline(
        [
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf", LogisticRegression()),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        df[feature],
        df[predict],
        test_size=0.20,
        random_state=0,
    )

    text_clf.fit(X_train, y_train)

    predicted = text_clf.predict(X_test)
    accuracy = np.mean(predicted == y_test)

    return text_clf, accuracy

def save_accuracy(model_id, accuracy):
    local_path = "/tmp/data.json"
    dict_accuracy = {'accuracy': accuracy}
    with open(local_path, "w") as file:
        json.dump(dict_accuracy, file)
    with open(local_path, "rb") as f:
        return bucket.upload_fileobj(f, model_id + '.json')

def handler(event, context):

    file_name = event['Records'][0]['s3']['object']['key']

    model_id = file_name.split('.')[0]
    
    df = get_DataFrame_s3(file_name)

    model, accuracy = train_model(df)

    save_accuracy(model_id,accuracy)
    save_model(model,model_id)

    return {"statusCode": 200,}
