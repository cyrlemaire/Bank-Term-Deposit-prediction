import pandas as pd
import numpy as np
import pickle

from subscription_forecast.config.config import read_yaml


# read config file:

CONFIG = read_yaml()
MODEL_NAME = CONFIG['model']['name']
THRESHOLD = CONFIG['model']['threshold']

data_path = CONFIG['predictions']['data_path']
dataset_filename = CONFIG['preprocessing']['dataset_filename']
delimiter = CONFIG['preprocessing']['csv_delimiter']

# Load model from pickle:


def load_model():
    with open("/Users/cyrillemaire/Documents/Yotta/Project/FINAL_TEST/productsubscription_dc_cl_js/subscription_forecast/finalized_model.sav", 'rb') as f:
        return pickle.load(f)


#final_pipeline = load_model()


def main():

    # get preprocessed dataset:

    prediction_data_full = pd.read_csv(data_path + '/' + dataset_filename, delimiter=delimiter)

    # get prediction probabilities:

    engineered_data = final_pipeline.steps[0][1].transform(prediction_data_full)
    predictions_proba = final_pipeline.steps[1][1].predict_proba(engineered_data)

    # get predictions with given threshold:

    predictions = np.where(predictions_proba[:, 1] >= THRESHOLD, 1, 0)

    # create and store a dataframe with client indexes and predictions :

    predictions_output = pd.DataFrame(data=predictions, index=prediction_data_full.index, columns=['predictions'])
    predictions_output.to_csv(CONFIG['predictions']['data_path']+'/'+CONFIG['predictions']['predictions_file_name'],
                              index=True)
    return predictions_output


if __name__ == '__main__':

    final_pipeline = load_model()

    main()

