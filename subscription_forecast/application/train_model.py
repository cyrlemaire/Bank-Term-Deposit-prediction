import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

from subscription_forecast.config.config import read_yaml
from subscription_forecast.domain import feature_engineering
from subscription_forecast.domain.model_evaluation import ModelEvaluator

# read config file:

CONFIG = read_yaml()

# get some config paramters

data_path = CONFIG['data']['data_path']
dataset_filename = CONFIG['preprocessing']['dataset_filename']
delimiter = CONFIG['preprocessing']['csv_delimiter']
MODEL_NAME = CONFIG['model']['name']
model_filename = CONFIG['model']['filename']
TARGET = CONFIG['filters']['TARGET']
features_to_drop = CONFIG['filters']['features_to_drop']


# model training pipeline:
"""Model names are:
    -'rf' for Random Forest
    -'lr' for Logistic Regression"""

if MODEL_NAME == "rf":
    optimal_parameters = {'rf_n_estimators': 668,
                          'rf_max_depth': 10,
                          'rf_min_samples_leaf': 10,
                          'rf_min_samples_split': 31,
                          'rf_bootstrap': True,
                          'rf_max_features': 'auto'}
    final_pipeline = Pipeline(steps=[
        ('transformer', feature_engineering.transformer),
        (MODEL_NAME, RandomForestClassifier(random_state=12,
                                            max_depth=optimal_parameters['rf_max_depth'],
                                            n_estimators=optimal_parameters['rf_n_estimators'],
                                            min_samples_leaf=optimal_parameters['rf_min_samples_leaf'],
                                            min_samples_split=optimal_parameters['rf_min_samples_split'],
                                            bootstrap=optimal_parameters['rf_bootstrap'],
                                            max_features=optimal_parameters['rf_max_features'])
         )])
elif MODEL_NAME == "lr":
    final_pipeline = Pipeline(steps=[
        ('transformer', feature_engineering.transformer),
        (MODEL_NAME, LogisticRegression(C=5, max_iter=500))
    ])
else:
    raise KeyError("wrong model name, try 'lr' or 'rf'")


if __name__ == '__main__':

    # get preprocessed dataset:

    client_full = pd.read_csv(data_path + '/' + dataset_filename, delimiter=delimiter)

    # split the data into train set and test set:

    y = client_full[TARGET]
    x = client_full.drop(columns=TARGET)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

    # fit and save model

    final_pipeline.fit(x_train, y_train)

    pickle.dump(final_pipeline, open(model_filename, 'wb'))

    # model performance evaluation

    evaluator = ModelEvaluator(MODEL_NAME, final_pipeline)

    evaluator.plot_precision_recall(x_test, y_test, x_train, y_train)


