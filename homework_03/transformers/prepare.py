import pandas as pd

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

@transformer
def transform(
    df: pd.DataFrame, **kwargs
) -> pd.DataFrame:
    df.tpep_dropoff_datetime = pd.to_datetime(df.tpep_dropoff_datetime)
    df.tpep_pickup_datetime = pd.to_datetime(df.tpep_pickup_datetime)
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60
    print("Length before filtering:", len(df))
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    print("Length after filtering:", len(df))
    categorical = ['PULocationID', 'DOLocationID']
    print('I passed this')
    df[categorical] = df[categorical].astype(str)
    print('I passed this too')
    
    return df