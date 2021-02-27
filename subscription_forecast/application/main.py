from subscription_forecast.infrastructure import preprocessing
from subscription_forecast.domain import plot_test
from subscription_forecast.application import data_profiling


# config variables:

data_path = '/Users/cyrillemaire/Documents/Yotta/Project/data/train'
customer_data_file_name = 'data.csv'
socio_eco_file_name = 'socio_eco.csv'

output_file_path = '/Users/cyrillemaire/Documents/Yotta/Project/EDA'
output_file_name = 'data_profiling_EDA.html'

columns_to_drop = ['duration_contact', 'contact' ]

if __name__ == '__main__':

    # load data
    customer = preprocessing.load_data_from(data_path, customer_data_file_name)
    socio_eco = preprocessing.load_data_from(data_path, socio_eco_file_name)
    socio_eco = preprocessing.socio_eco_imputer(socio_eco)

    # merge data
    customer_full = preprocessing.link_dataframes(customer, socio_eco)
    print(customer_full.info())

    # drop feature test
    FeatureRemover = preprocessing.FeatureDrop(columns_to_drop)
    customer_full = FeatureRemover.transform(customer_full)
    print(customer_full.info())


