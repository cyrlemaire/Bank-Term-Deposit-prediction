from subscription_forecast.infrastructure import preprocessing
from subscription_forecast.domain import plot_test
from subscription_forecast.application import data_profiling





# transform feature for ML model fitting
client_full = preprocessing.transformer.fit_transform(client_full)
print(client_full)


if __name__ == '__main__':
    breakpoint()


