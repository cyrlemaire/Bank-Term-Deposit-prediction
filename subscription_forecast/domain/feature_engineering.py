import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


# config variables:

features_to_indicator = ['has_perso_loan', 'has_housing_loan', 'has_default', 'result_last_campaign']
socio_eco_features = ['employment_variation_rate', 'idx_consumer_price', 'idx_consumer_confidence']
numeric_features = ['balance', 'nb_day_last_contact', 'nb_contact']


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


class AgeImputer(BaseEstimator, TransformerMixin):
    """replace the '123' age values by the median age"""
    def __init__(self):
        pass

    def fit(self, data_x, y=None):
        return self

    def transform(self, data_x, y=None):
        data_x['age'] = data_x['age'].replace({123: np.NaN})
        data_x['age'] = data_x['age'].fillna(float(data_x['age'].median()))
        return data_x


class JobImputer(BaseEstimator, TransformerMixin):
    """create an 'unknown' category for missing job values"""
    def __init__(self):
        pass

    def fit(self, data_x, y=None):
        return self

    def transform(self, data_x, y=None):
        data_x['job_type'] = data_x['job_type'].replace({np.NaN: 'unknown'})

        return data_x

# pipelines to transform dataset


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







