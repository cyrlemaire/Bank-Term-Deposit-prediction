from subscription_forecast.infrastructure import load_data
from subscription_forecast.domain import plot_test
from subscription_forecast.application import data_profiling


data_path = '/Users/cyrillemaire/Documents/Yotta/Project/data/train'
customer_data_file_name = 'data.csv'
socio_eco_file_name = 'socio_eco.csv'

output_file_path = '/Users/cyrillemaire/Documents/Yotta/Project/EDA'
output_file_name = 'data_profiling_EDA.html'

if __name__ == '__main__':

    customer_data = load_data.load_data_from(data_path, customer_data_file_name)
    socio_eco_data = load_data.load_data_from(data_path, socio_eco_file_name)
    print(customer_data.head())

    plot_test.boxplot(customer_data, "SUBSCRIPTION", "NB_DAY_LAST_CONTACT")

    data_profiling.data_profiling_from(customer_data, output_file_path, output_file_name)

