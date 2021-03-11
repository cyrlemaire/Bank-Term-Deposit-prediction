from subscription_forecast.application.train_model import final_pipeline, x_test, y_test
from subscription_forecast.config.config import read_yaml
from subscription_forecast.infrastructure import preprocessing

from sklearn.metrics import recall_score,\
    precision_score, \
    confusion_matrix, \
    accuracy_score

import pandas as pd
import numpy as np

# read config file:

CONFIG = read_yaml()

# get filters for feature engineering:

TARGET = CONFIG['filters']['TARGET']
features_to_drop = CONFIG['filters']['features_to_drop']
features_to_indicator = CONFIG['filters']['features_to_indicator']
socio_eco_features = CONFIG['filters']['socio_eco_features']
numeric_features = CONFIG['filters']['numeric_features']
categorical_features = CONFIG['filters']['categorical_features']

# get data for prediction:
prediction_data_full = preprocessing.features_from(CONFIG['prediction_data']['data_path'],
                                          CONFIG['prediction_data']['client_file_name'],
                                          CONFIG['prediction_data']['socio_eco_file_name'],
                                          features_to_drop)

prediction_data = x_test
y = y_test
# get prediction probabilities:
engineered_data = final_pipeline.steps[0][1].transform(prediction_data)
predictions_proba = final_pipeline.steps[1][1].predict_proba(engineered_data)

# get predictions with given threshold:
THRESHOLD = CONFIG['model']['threshold']
predictions = np.where(predictions_proba[:, 1] >= THRESHOLD, 1, 0)

print("Model accuracy : ", np.around(accuracy_score(y, predictions), decimals=3))
print("Model precision : ", np.around(precision_score(y, predictions, average="binary", pos_label=1), decimals=3))
print("Model recall = ", np.around(recall_score(y, predictions, average="binary", pos_label=1), decimals=3))
print("Confusion Matrix : \n", confusion_matrix(y, predictions, labels=[1, 0]))

# create and store a dataframe with client indexes and predictions :

predictions_output = pd.DataFrame(data=predictions, index=x_test.index, columns=['predictions'])
predictions_output.to_csv(CONFIG['prediction_data']['data_path']+'/'+'predictions.csv')