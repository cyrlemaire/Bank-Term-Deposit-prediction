import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler

from subscription_forecast.config.config import read_yaml

# read config file:

CONFIG = read_yaml()

# get filters for feature engineering:

features_to_indicator = CONFIG['filters']['features_to_indicator']
socio_eco_features = CONFIG['filters']['socio_eco_features']
numeric_features = CONFIG['filters']['numeric_features']
categorical_features = CONFIG['filters']['categorical_features']


# transformers:

class IndicatorTransformer(BaseEstimator, TransformerMixin):
    """transform a subset categorical feature into a 1/0 feature"""
    def __init__(self):
        pass

    def fit(self, data_x, y=None):
        return self

    def transform(self, data_x, y=None):
        data_x = np.where((data_x == 'Yes') | (data_x == 'Succes'), 1, 0)
        return data_x


class DateTransformer(BaseEstimator, TransformerMixin):
    """Extract weekdays and months from date and do target encoding"""
    def __init__(self):
        pass

    def fit(self, data_x, y):
        """ create mask for target encoding. 'encoding' is a
         dictionary where each category is paired with with its
         corresponding numeric value"""
        data_x['date'] = pd.to_datetime(data_x['date'], infer_datetime_format=True)
        data_x['month'] = data_x['date'].dt.month_name()
        data_x_y = pd.concat((data_x, y.rename('y')), axis=1)
        encoding_month = data_x_y.groupby('month').agg({'y': 'mean'}, index='month')
        self.encoding_month = encoding_month
        return self

    def transform(self, data_x):
        data_x['date'] = pd.to_datetime(data_x['date'], infer_datetime_format=True)
        data_x['month'] = data_x['date'].dt.month_name()
        data_x.drop(columns=['date'], inplace=True)
        data_x['month'] = data_x['month'].replace(self.encoding_month['y'].to_dict())
        return data_x


class AgeImputer(BaseEstimator, TransformerMixin):
    """replace the '123' age values by the median age"""
    def __init__(self):
        pass

    def fit(self, data_x, y=None):
        return self

    def transform(self, data_x, y=None):
        data_x['age'] = data_x['age'].replace({123: np.NaN})
        data_x['age'] = data_x['age'].fillna(data_x['age'].median())
        return data_x


class JobTransformer(BaseEstimator, TransformerMixin):
    """create an 'unknown' category for missing job values
    then do likelihood encoding for the categories"""
    def __init__(self):
        pass

    def fit(self, data_x, y):
        """ create mask for target encoding. 'encoding' is a
         dictionary where each category is paired with with its
         corresponding numeric value"""
        data_x['job_type'] = data_x['job_type'].replace({np.NaN: 'unknown'})
        data_x_y = pd.concat((data_x, y.rename('y')), axis=1)
        encoding = data_x_y.groupby('job_type').agg({'y': 'mean'}, index='job_type')
        self.encoding = encoding
        return self

    def transform(self, data_x):
        data_x['job_type'] = data_x['job_type'].replace({np.NaN: 'unknown'})
        data_x['job_type'] = data_x['job_type'].replace(self.encoding['y'].to_dict())
        return data_x


class DayLastContactTransformer(BaseEstimator, TransformerMixin):
    """ sets the -1's as 0 and transforms the >0 values as their reverse"""
    def __init__(self):
        pass

    def fit(self, data_x, y=None):
        return self

    def transform(self, data_x, y=None):
        data_x['nb_day_last_contact'] = np.where(data_x.nb_day_last_contact < 0, 0, 1/data_x.nb_day_last_contact)
        return data_x


class HousingPersoLoanTransformer(BaseEstimator, TransformerMixin):
    """creates a new column that target encode the number of loans the rows are having"""
    def __init__(self):
        pass

    def fit(self, data_x, y=None):
        return self

    def transform(self, data_x, y=None):
        data_x['has_housing_loan'] = data_x['has_housing_loan'].replace({'Yes': 1, 'No': 0})
        data_x['has_perso_loan'] = data_x['has_perso_loan'].replace({'Yes': 1, 'No': 0, np.NaN: 0})
        data_x['has_loan'] = data_x['has_housing_loan'] + data_x['has_perso_loan']
        return data_x


# custom transformer pipelines

age_transformer = Pipeline(steps=[
    ('age_imputer', AgeImputer()),
    ('age_scaler', RobustScaler())])

categorical_transformer = Pipeline(steps=[
    ('categorical_imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
    ('categorical_transformer', OneHotEncoder(drop='first'))])

day_last_contact_transformer = Pipeline(steps=[
    ('day_last_contact_transformer', DayLastContactTransformer()),
    ('day_last_contact_scaler', RobustScaler())])

# Feature engineering pipeline:

transformer = ColumnTransformer(
    transformers=[('feature_indicator', IndicatorTransformer(), features_to_indicator),
                  ('age_transformer', age_transformer, ['age']),
                  ('date_transformer', DateTransformer(), ['date']),
                  ('numeric_scaler', RobustScaler(), numeric_features),
                  ('category_transformer', categorical_transformer, categorical_features),
                  ('socio_eco_scaler', RobustScaler(), socio_eco_features),
                  ('housing_perso_loan_transformer', HousingPersoLoanTransformer(), ['has_housing_loan', 'has_perso_loan']),
                  ('day_last_contact_transformer', day_last_contact_transformer, ['nb_day_last_contact'])
                  ],
    remainder='drop'
)
