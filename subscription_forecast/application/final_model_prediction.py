import pandas as pd
import numpy as np
import pickle

from subscription_forecast.config.config import read_yaml
from subscription_forecast.infrastructure import preprocessing

# read config file:

CONFIG = read_yaml()
MODEL_NAME = CONFIG['model']['name']

# get filters for feature engineering:

features_to_drop = CONFIG['filters']['features_to_drop']

# Load model from pickle:

def load_model():
    with open("/Users/cyrillemaire/Documents/Yotta/Project/FINAL_TEST/productsubscription_dc_cl_js/subscription_forecast/finalized_model.sav", 'rb') as f:
        return pickle.load(f)

final_pipeline = load_model()


def main():

    # get data for prediction:

    prediction_data_full = preprocessing.features_from(CONFIG['prediction_data']['data_path'],
                                                       CONFIG['prediction_data']['client_file_name'],
                                                       CONFIG['prediction_data']['socio_eco_file_name'],
                                                       features_to_drop)

    # get prediction probabilities:

    engineered_data = final_pipeline.steps[0][1].transform(prediction_data_full)
    predictions_proba = final_pipeline.steps[1][1].predict_proba(engineered_data)

    # get predictions with given threshold:

    THRESHOLD = CONFIG['model']['threshold']
    predictions = np.where(predictions_proba[:, 1] >= THRESHOLD, 1, 0)

    # create and store a dataframe with client indexes and predictions :

    predictions_output = pd.DataFrame(data=predictions, index=prediction_data_full.index, columns=['predictions'])
    predictions_output.to_csv(CONFIG['prediction_data']['data_path']+'/'+CONFIG['prediction_data']['predictions_file_name'],
                              index=True)
    return predictions_output

if __name__ == '__main__':
    main()