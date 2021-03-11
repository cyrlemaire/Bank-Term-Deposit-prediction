from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler

from subscription_forecast.config.config import read_yaml
from subscription_forecast.infrastructure import preprocessing
from subscription_forecast.domain import feature_engineering
from subscription_forecast.domain.model_evaluation import ModelEvaluator

# read config file:

CONFIG = read_yaml()

# get filters for feature engineering:

TARGET = CONFIG['filters']['TARGET']
features_to_drop = CONFIG['filters']['features_to_drop']
features_to_indicator = CONFIG['filters']['features_to_indicator']
socio_eco_features = CONFIG['filters']['socio_eco_features']
numeric_features = CONFIG['filters']['numeric_features']
categorical_features = CONFIG['filters']['categorical_features']

MODEL_NAME = CONFIG['model']['name']

# get preprocessed dataset:

client_full = preprocessing.features_from(CONFIG['data']['data_path'],
                                          CONFIG['data']['client_file_name'],
                                          CONFIG['data']['socio_eco_file_name'],
                                          features_to_drop)

# split the data into train set and test set:

y = client_full[TARGET]
y = y.replace({'Yes': 1, 'No': 0})
x = client_full.drop(columns=TARGET)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

# Feature engineering pipeline:

transformer = ColumnTransformer(
    transformers=[('feature_indicator', feature_engineering.IndicatorTransformer(), features_to_indicator),
                  ('age_transformer', feature_engineering.age_transformer, ['age']),
                  ('date_transformer', feature_engineering.DateTransformer(), ['date']),
                  ('numeric_scaler', RobustScaler(), numeric_features),
                  ('category_transformer', feature_engineering.categorical_transformer, categorical_features),
                  ('socio_eco_scaler', RobustScaler(), socio_eco_features),
                  ('housing_perso_loan_transformer', feature_engineering.HousingPersoLoanTransformer(), ['has_housing_loan', 'has_perso_loan']),
                  ('day_last_contact_transformer', feature_engineering.day_last_contact_transformer, ['nb_day_last_contact'])
                  ],
    remainder='drop'
)

# model training pipeline:
"""Model names are:
    -'rf' for Random Forest
    -'lr' for Logistic Regression
    -'svm' for Support Vector Machine"""

# TODO Regressor factory

if MODEL_NAME == "rf":
    final_pipeline = Pipeline(steps=[
        ('transformer', transformer),
        (MODEL_NAME, RandomForestClassifier(n_estimators=200, max_depth=12, random_state=12))
    ])
elif MODEL_NAME == "lr":
    final_pipeline = Pipeline(steps=[
        ('transformer', transformer),
        (MODEL_NAME, LogisticRegression(C=5, max_iter=500))
    ])
else:
    raise KeyError("wrong model name, try 'lr' or 'rf'")

final_pipeline.fit(x_train, y_train)

# model performance evaluation

#evaluator = ModelEvaluator(MODEL_NAME, final_pipeline)

#evaluator.print_metrics(x_test, y_test, x_train, y_train)
#evaluator.plot_precision_recall(x_test, y_test, x_train, y_train, transformer)

