import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder



# config variables:

features_to_indicator = ['has_perso_loan', 'has_housing_loan']
socio_eco_features = ['employment_variation_rate', 'idx_consumer_price', 'idx_consumer_confidence']
numeric_features = ['balance', 'nb_contact']
categorical_features = ['status', 'education']

# transformers:


class IndicatorTransformer(BaseEstimator, TransformerMixin):
    """transform a subset categorical feature into a 1/0 feature"""
    def __init__(self):
        pass

    def fit(self, data_x, y=None):
        return self

    def transform(self, data_x, y=None):
        data_x = np.where(data_x == ('Yes' or 'Succes'), 1, 0)
        return data_x


class DateTransformer(BaseEstimator, TransformerMixin):
    """Extract weekdays and months from date and do targeet encoding"""
    def __init__(self):
        pass

    def fit(self, data_x, y):
        """ create mask for target encoding. 'encoding' is a
         dictionary where each category is paired with with its
         corresponding numeric value"""

        data_x['date'] = pd.to_datetime(data_x['date'], infer_datetime_format=True)
        data_x['weekday'] = data_x['date'].dt.day_name()
        data_x['month'] = data_x['date'].dt.month_name()
        data_x_y = pd.concat((data_x, y.rename('y')), axis=1)
        encoding_weekday = data_x_y.groupby('weekday').agg({'y': 'mean'}, index='weekday')
        encoding_month = data_x_y.groupby('month').agg({'y': 'mean'}, index='month')
        self.encoding_weekday = encoding_weekday
        self.encoding_month = encoding_month
        return self

    def transform(self, data_x):
        data_x['date'] = pd.to_datetime(data_x['date'], infer_datetime_format=True)
        data_x['weekday'] = data_x['date'].dt.day_name()
        data_x['month'] = data_x['date'].dt.month_name()
        data_x.drop(columns=['date'], inplace=True)
        data_x['weekday'] = data_x['weekday'].replace(self.encoding_weekday['y'].to_dict())
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
        data_x['age'] = data_x['age'].fillna(float(data_x['age'].median()))
        data_x['age2'] = data_x['age']**2
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


# pipelines to transform dataset


age_transformer = Pipeline(steps=[
    ('age_imputer', AgeImputer()),
    ('age_scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('categorical_imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
    ('categorical_transformer', OneHotEncoder(drop='first'))])

transformer = ColumnTransformer(
    transformers=[('feature_indicator', IndicatorTransformer(), features_to_indicator),
                  ('age_transformer', age_transformer, ['age']),
                 # ('job_transformer', JobTransformer(), ['job_type']),
                  ('date_transformer', DateTransformer(), ['date']),
                  ('numeric_scaler', StandardScaler(), numeric_features),
                  ('category_transformer', categorical_transformer, categorical_features),
                  ('socio_eco_scaler', StandardScaler(), socio_eco_features)],
    remainder='drop'
)







