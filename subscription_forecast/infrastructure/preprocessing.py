import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion



def load_data_from(data_path: str, filename: str) -> pd.DataFrame:
    """ load a dataset from in a .csv file and return a dataframe
    with lower case column names"""
    data = pd.read_csv(data_path+'/'+filename, delimiter=';')
    data.columns = data.columns.str.lower()
    return data


def socio_eco_imputer(socio_eco):
    """Fills the NaN values specifically for the socio_eco dataset."""
    socio_eco_complete = socio_eco.copy()
    socio_eco_complete.iloc[[4, 8, 9, 20, 21], 1] = [-0.1, -0.2, -0.2, -3.0, 3.0]
    socio_eco_complete = socio_eco_complete.interpolate()
    return socio_eco_complete


def link_dataframes(data_left, data_right):
    """Extract month and year from a date column in 2 dataframes,
    use this truncated date as key for a left join between the two datasets"""
    data_left['date_trunc'] = data_left['date'].str.slice(stop=7)
    data_right['date_trunc'] = data_right['date'].str.slice(stop=7)
    data_left = (pd.merge(data_left, data_right, left_on='date_trunc', right_on='date_trunc', how='left'))
    data_left.drop('date_trunc', axis=1, inplace=True)
    data_left.drop('date_y', axis=1, inplace=True)
    data_left.rename(columns={'date_x': 'date'}, inplace=True)
    return data_left


class FeatureDrop(BaseEstimator, TransformerMixin):
    """remove a subset of features"""
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.drop(self.columns_to_drop, axis=1)
        return X

