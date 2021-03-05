from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from subscription_forecast.infrastructure import preprocessing
from subscription_forecast.domain import feature_engineering

# some parameters to put in config
data_path = '/Users/cyrillemaire/Documents/Yotta/Project/data/train'
client_data_file_name = 'data.csv'
socio_eco_file_name = 'socio_eco.csv'

target = 'subscription'

# model type

model = "random_forest"

# get data to feed into model pipeline

client_full = preprocessing.features_from(data_path, client_data_file_name, socio_eco_file_name)

y = client_full['subscription']
X = client_full.drop(columns = target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# model pipeline

final_pipeline = Pipeline(steps=[
    ('transformer', feature_engineering.transformer),
    ('rf_estimator', RandomForestClassifier(n_estimators=100))
])

final_pipeline.fit(X_train, y_train)


