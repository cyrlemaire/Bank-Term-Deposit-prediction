import pandas as pd


def load_data_from(data_path: str, filename: str) -> pd.DataFrame:

    data = pd.read_csv(data_path+'/'+filename,
                                  delimiter=';')
    return data
