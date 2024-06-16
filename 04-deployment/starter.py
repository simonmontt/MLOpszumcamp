#!/usr/bin/env python
# coding: utf-8


#get_ipython().system('pip freeze | grep scikit-learn')




#get_ipython().system('pip install pyarrow')



#get_ipython().system('python -V')


import pickle
import pandas as pd



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


df = read_data('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet')


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


print(f' The standard deviation of the predicted duration for this dataset is: {y_pred.std()}')


# Prompt the user to enter the year
year = input("Please enter the year: ")

# Prompt the user to enter the month
month = input("Please enter the month: ")

# Convert inputs to integers
year = int(year)
month = int(month)


df['ride_id'] = f'{2023:04d}/{3:02d}_' + df.index.astype('str')


# In[14]:


output_file = f'yellow/{year:04d}-{month:02d}.parquet'


# In[20]:


def save_results(df, y_pred, output_file):
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    return df_result

df_mod = save_results(df, y_pred, output_file)

#df_mod.head(5)

df_mod.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)

print(f' The mean of the predicted duration for this dataset is: {y_pred.mean()}')