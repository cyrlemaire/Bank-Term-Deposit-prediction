from subscription_forecast.infrastructure import preprocessing

import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

# config variables:

data_path = '/Users/cyrillemaire/Documents/Yotta/Project/data/train'
client_data_file_name = 'data.csv'
socio_eco_file_name = 'socio_eco.csv'


features_to_indicator = ['has_perso_loan', 'has_housing_loan', 'has_default', 'result_last_campaign']
socio_eco_features = ['employment_variation_rate', 'idx_consumer_price', 'idx_consumer_confidence']
numeric_features = ['balance', 'nb_day_last_contact', 'nb_contact']

target = 'subscription'

# transformers:


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


class JobImputer(BaseEstimator, TransformerMixin):
    """transform a subset categorical feature into a 1/0 feature"""
    def __init__(self):
        print('job imputer')

    def fit(self, data_x, y=None):
        return self

    def transform(self, data_x, y=None):
        data_x['job_type'] = data_x['job_type'].replace({np.NaN: 'unknown'})

        return data_x


age_transformer = Pipeline(steps=[
    ('age_imputer', AgeImputer()),
    ('age_scaler', StandardScaler())])

job_transformer = Pipeline(steps=[
    ('job_imputer', JobImputer()),
    ('job_encoder', OneHotEncoder())])

transformer = ColumnTransformer(
    transformers=[('feature_indicator', IndicatorTransformer(), features_to_indicator),
                  ('age_transformer', age_transformer, ['age']),
                  ('job_transformer', job_transformer, ['job_type']),
                  ('numeric_scaler', StandardScaler(), numeric_features),
                  ('socio_eco_scaler', StandardScaler(), socio_eco_features)],
    remainder='drop'
)

# transform feature to feed in pipelinee

client_full = preprocessing.features_from(data_path, client_data_file_name, socio_eco_file_name)

y = client_full['subscription']
X = client_full.drop(columns = target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

client_full = transformer.fit_transform(client_full)

#model


final_pipeline = Pipeline(steps = [
    ('transformer', transformer),
    ('rf_estimator', RandomForestClassifier())
])

final_pipeline.fit(X_train, y_train)

print(final_pipeline.score(X_test, y_test))

y_test = y_test.to_numpy()
y_pred = final_pipeline.predict(X_test)
print("Precision = ", precision_score(y_test, y_pred, average="binary", pos_label='Yes'))
print("Recall = ", recall_score(y_test, y_pred, average="binary", pos_label='Yes'))


