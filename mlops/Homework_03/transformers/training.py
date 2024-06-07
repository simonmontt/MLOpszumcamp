from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from typing import List, Tuple
import pandas as pd


if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

@transformer
def train_model(df: pd.DataFrame, 
) -> Tuple[DictVectorizer, LinearRegression]:
    """
    Train a linear regression model using the given DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the dataset.

    Returns:
        Tuple[DictVectorizer, LinearRegression]: Trained DictVectorizer and Linear Regression model.
    """
    print("1")
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    print("2")
    categorical = ['PULocationID', 'DOLocationID']
    print("3")
    numerical = ['trip_distance']
    print("4")
    train_dicts = df[categorical].to_dict(orient='records')
    print("5")
    dv = DictVectorizer()
    print(dv)
    X_train = dv.fit_transform(train_dicts)
    print("6")
    target = 'duration'
    print("7")
    y_train = df[target].values
    print("8")
    lr = LinearRegression()
    print("9")
    lr.fit(X_train, y_train)
    print("10")
    print(lr.intercept_)
    return dv, lr
