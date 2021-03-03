import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


# some parameters to put in config
features_to_drop = ['contact', 'duration_contact', 'status', 'education', 'nb_contact_last_campaign']
features_to_indicator = ['has_perso_loan', 'has_housing_loan', 'has_default', 'result_last_campaign']
socio_eco_features = ['employment_variation_rate', 'idx_consumer_price', 'idx_consumer_confidence']
numeric_features = ['balance', 'age', 'nb_day_last_contact', 'nb_contact']
target = ['subscription']

# config variables:

data_path = '/Users/cyrillemaire/Documents/Yotta/Project/data/train'
client_data_file_name = 'data.csv'
socio_eco_file_name = 'socio_eco.csv'

output_file_path = '/Users/cyrillemaire/Documents/Yotta/Project/EDA'
output_file_name = 'data_profiling_EDA.html'


def train_test_from(data_path: str, client_data_file_name: str, socio_eco_file_name: str):
    """extract the 2 csv files and create a dataset to feed in the
    Feature engineering pipeline"""

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

    def set_split(full_data):
        y = full_data[target]
        X = full_data.drop[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    # load data
    client = load_data_from(data_path, client_data_file_name)
    socio_eco = load_data_from(data_path, socio_eco_file_name)
    socio_eco = socio_eco_imputer(socio_eco)

    # merge data
    client_full = link_dataframes(client, socio_eco)

    #set_split:
    X_train, X_test, y_train, y_test = set_split(client_full)

    return X_train, X_test, y_train, y_test


class IndicatorTransformer(BaseEstimator, TransformerMixin):
    """transform a subset categorical feature into a 1/0 feature"""
    def __init__(self):
        print('Indicator transform')

    def fit(self, data_x, y=None):
        return self

    def transform(self, data_x, y=None):
        data_x = np.where(data_x == ('Yes' or 'Succes'), 1, 0)
        return data_x

class AgeImputer(BaseEstimator, TransformerMixin):
    """transform a subset categorical feature into a 1/0 feature"""
    def __init__(self):
        print('age imputer')

    def fit(self, data_x, y=None):
        return self

    def transform(self, data_x, y=None):
        data_x['age'] = data_x['age'].replace({123: np.NaN})
        data_x['age'] = data_x['age'].fillna(float(data_x['age'].median()))
        return data_x


age_transformer = Pipeline(steps=[
    ('age_imputer', AgeImputer()),
    ('age_scaler', StandardScaler())])

transformer = ColumnTransformer(
    transformers=[('feature_indicator', IndicatorTransformer(), features_to_indicator),
                  ('age_transformer', age_transformer, ['age']),
                  ('job_encoder', OneHotEncoder(), ['job_type']),
                  ('socio_eco_scaler', StandardScaler(), socio_eco_features)],
    remainder='drop'
)



