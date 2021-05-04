import pandas as pd

from subscription_forecast.config.config import read_yaml

# read config file:

CONFIG = read_yaml()

features_to_drop = CONFIG['filters']['features_to_drop']
data_path = CONFIG['training_data']['data_path']
client_data_file_name = CONFIG['training_data']['client_file_name']
socio_eco_file_name = CONFIG['training_data']['socio_eco_file_name']


class Dataset:
    """Create dataset model training"""
    @staticmethod
    def load_data_from(data_path: str, filename: str) -> pd.DataFrame:
        """ load a dataset from in a .csv file and return a dataframe
        with lower case column names"""
        data = pd.read_csv(data_path+'/'+filename, delimiter=CONFIG['preprocessing']['csv_delimiter'])
        data.columns = data.columns.str.lower()
        return data

    @staticmethod
    def socio_eco_imputer(socio_eco: pd.Dataframe) -> pd.DataFrame:
        """Fills the NaN values specifically for the socio_eco dataset."""
        socio_eco_complete = socio_eco.copy()
        socio_eco_complete = socio_eco_complete.interpolate()
        return socio_eco_complete

    @staticmethod
    def link_dataframes(data_left: pd.Dataframe, data_right: pd.Dataframe) -> pd.DataFrame:
        """Extract month and year from a date column in 2 dataframes,
        use this truncated date as key for a left join between the two datasets"""
        data_left['date_trunc'] = data_left['date'].str.slice(stop=7)
        data_right['date_trunc'] = data_right['date'].str.slice(stop=7)
        data_left = (pd.merge(data_left, data_right, left_on='date_trunc', right_on='date_trunc', how='left'))
        data_left.drop('date_trunc', axis=1, inplace=True)
        data_left.drop('date_y', axis=1, inplace=True)
        data_left.rename(columns={'date_x': 'date'}, inplace=True)
        return data_left

    @staticmethod
    def drop_features(full_data: pd.Dataframe, to_drop: list) -> pd.Dataframe:
        """Drop useless columns"""
        full_data = full_data.drop(columns=to_drop)
        return full_data

    @staticmethod
    def drop_nan(full_data: pd.Dataframe) -> pd.Dataframe:
        """drop rows with more than 2 NaN
        => not used in optimized model"""
        print(f"Before preprocessing the dataset contains {full_data.isna().sum().sum()} missing values")
        rows_removed = full_data
        full_data = full_data.dropna(thresh=full_data.shape[1]-2, axis=0)
        rows_removed = rows_removed.drop(full_data.index.values.tolist(), axis=0)
        print(f"After preprocessing the dataset contains {full_data.isna().sum().sum()} missing values")
        return full_data, rows_removed


if __name__ == '__main__':

    # load data
    client = Dataset.load_data_from(data_path, client_data_file_name)
    socio_eco = Dataset.load_data_from(data_path, socio_eco_file_name)
    socio_eco = Dataset.socio_eco_imputer(socio_eco)

    # merge data
    client_full = Dataset.link_dataframes(client, socio_eco)

    # drop features:
    client_full = Dataset.drop_features(client_full,features_to_drop)






