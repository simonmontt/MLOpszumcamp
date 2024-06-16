#!/usr/bin/env python
# coding: utf-8

import argparse
import pickle
import pandas as pd

# Load the model
with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

# Read data from the provided URL
df = read_data('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet')

dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)

print(f'The standard deviation of the predicted duration for this dataset is: {y_pred.std()}')

# Use argparse to read the year and month from command line arguments
parser = argparse.ArgumentParser(description='Process year and month.')
parser.add_argument('--year', type=int, required=True, help='Year to process')
parser.add_argument('--month', type=int, required=True, help='Month to process')

args = parser.parse_args()

year = args.year
month = args.month

df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

output_file = f'{year:04d}-{month:02d}.parquet'

def save_results(df, y_pred, output_file):
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    return df_result

df_mod = save_results(df, y_pred, output_file)

df_mod.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)

print(f'The mean of the predicted duration for this dataset is: {y_pred.mean()}')
