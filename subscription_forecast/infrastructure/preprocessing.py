import pandas as pd


# pre-processing functions


def features_from(data_path: str, client_data_file_name: str, socio_eco_file_name: str, to_drop: list):
    """extract the 2 csv files and create a dataset to feed in the
    Feature engineering pipeline"""

    def load_data_from(data_path: str, filename: str) -> pd.DataFrame:
        """ load a dataset from in a .csv file and return a dataframe
        with lower case column names"""
        data = pd.read_csv(data_path+'/'+filename, delimiter=';')
        data.columns = data.columns.str.lower()
        return data

    def socio_eco_imputer(socio_eco):
        """Fills the NaN values specifically for the socio_eco dataset."""
        socio_eco_complete = socio_eco.copy()
        socio_eco_complete.iloc[[4, 8, 9, 20, 21], 1] = [-0.1, -0.2, -0.2, -3.0, 3.0]
        socio_eco_complete = socio_eco_complete.interpolate()
        return socio_eco_complete

    def link_dataframes(data_left, data_right):
        """Extract month and year from a date column in 2 dataframes,
        use this truncated date as key for a left join between the two datasets"""
        data_left['date_trunc'] = data_left['date'].str.slice(stop=7)
        data_right['date_trunc'] = data_right['date'].str.slice(stop=7)
        data_left = (pd.merge(data_left, data_right, left_on='date_trunc', right_on='date_trunc', how='left'))
        data_left.drop('date_trunc', axis=1, inplace=True)
        data_left.drop('date_y', axis=1, inplace=True)
        data_left.rename(columns={'date_x': 'date'}, inplace=True)
        return data_left

    def drop_features(full_data):
        full_data = full_data.drop(columns=to_drop)
        return full_data

    def drop_nan(full_data):
        """drop rows with more than 1 NaN"""
        rows_removed = full_data
        full_data = full_data.dropna(thresh=full_data.shape[1]-2, axis=0)
        rows_removed = rows_removed.drop(full_data.index.values.tolist(), axis=0)
        return full_data, rows_removed

    # load data
    client = load_data_from(data_path, client_data_file_name)
    socio_eco = load_data_from(data_path, socio_eco_file_name)
    socio_eco = socio_eco_imputer(socio_eco)

    # merge data
    client_full = link_dataframes(client, socio_eco)

    # drop features:

    client_full = drop_features(client_full)
    print(f"Before preprocessing the dataset contains {client_full.isna().sum().sum()} missing values")

    # drop nan:

    client_full, rows_removed = drop_nan(client_full)
    print(f"After preprocessing the dataset contains {client_full.isna().sum().sum()} missing values")
    return client_full, rows_removed






