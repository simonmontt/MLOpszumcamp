#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip freeze | grep scikit-learn')


# In[4]:


get_ipython().system('pip install pyarrow')


# In[5]:


get_ipython().system('python -V')


# In[6]:


import pickle
import pandas as pd


# In[7]:


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[8]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[9]:


df = read_data('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet')


# In[10]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# In[12]:


print(f' The standard deviation of the predicted duration for this dataset is: {y_pred.std()}')


# In[13]:


df['ride_id'] = f'{2023:04d}/{3:02d}_' + df.index.astype('str')


# In[14]:


output_file = f'yellow/{2023:04d}-{3:02d}.parquet'


# In[20]:


def save_results(df, y_pred, output_file):
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    return df_result

df_mod = save_results(df, y_pred, output_file)


# In[21]:


df_mod.head(5)


# In[22]:


df_mod.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)

