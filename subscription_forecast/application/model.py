from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from subscription_forecast.infrastructure import preprocessing
from subscription_forecast.domain import feature_engineering

from subscription_forecast.application.model_evaluation import ModelEvaluator

# some parameters to put in config
data_path = '/Users/cyrillemaire/Documents/Yotta/Project/data/train'
client_data_file_name = 'data.csv'
socio_eco_file_name = 'socio_eco.csv'

target = 'subscription'

# model type lr ou rf

MODEL_NAME = "lr"


# get data to feed into model pipeline

client_full = preprocessing.features_from(data_path, client_data_file_name, socio_eco_file_name)

y = client_full['subscription']
y = y.replace({'Yes': 1, 'No': 0})
x = client_full.drop(columns=target)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# model pipeline selection
# TODO factory Regressor

if MODEL_NAME == "rf":
    final_pipeline = Pipeline(steps=[
        ('transformer', feature_engineering.transformer),
        (MODEL_NAME, RandomForestClassifier(n_estimators=200, max_depth=8, verbose=False))
    ])
elif MODEL_NAME == "lr":
    final_pipeline = Pipeline(steps=[
        ('transformer', feature_engineering.transformer),
        (MODEL_NAME, LogisticRegression(C=0.1))
    ])
else:
    raise KeyError("wrong model name, try 'lr' or 'rf'")


final_pipeline.fit(x_train, y_train)

evaluator = ModelEvaluator(MODEL_NAME, final_pipeline)

evaluator.print_metrics(x_test, y_test, x_train, y_train)
evaluator.plot_precision_recall(x_test, y_test, x_train, y_train, feature_engineering.transformer)

