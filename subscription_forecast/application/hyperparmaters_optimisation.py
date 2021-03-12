import optuna
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from subscription_forecast.config.config import read_yaml
from subscription_forecast.infrastructure import preprocessing
from subscription_forecast.domain.feature_engineering import transformer


# read config file
CONFIG = read_yaml()

# get filters for feature engineering:

TARGET = CONFIG['filters']['TARGET']
features_to_drop = CONFIG['filters']['features_to_drop']
features_to_indicator = CONFIG['filters']['features_to_indicator']
socio_eco_features = CONFIG['filters']['socio_eco_features']
numeric_features = CONFIG['filters']['numeric_features']
categorical_features = CONFIG['filters']['categorical_features']

MODEL_NAME = CONFIG['model']['name']

# get preprocessed dataset:

client_full = preprocessing.features_from(CONFIG['training_data']['data_path'],
                                          CONFIG['training_data']['client_file_name'],
                                          CONFIG['training_data']['socio_eco_file_name'],
                                          features_to_drop)

# split the data into train set and test set:

y = client_full[TARGET]
y = y.replace({'Yes': 1, 'No': 0})
x = client_full.drop(columns=TARGET)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# get the feature engineered data:

transformer.fit(x_train, y_train)
X_FE = transformer.transform(x_train)


# Step 1. Define an objective function to be maximized.
def objective(trial):

    rf_n_estimators = trial.suggest_int("rf_n_estimators", 10, 1000)
    rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
    rf_min_samples_leaf = trial.suggest_int("rf_min_samples_leaf", 1, 10)
    rf_min_samples_split = trial.suggest_int("rf_min_samples_split", 2, 35)
    rf_bootstrap = trial.suggest_categorical("rf_bootstrap", [True, False])
    rf_max_features = trial.suggest_categorical("rf_max_features", ['auto', 'sqrt'])
    classifier_obj = RandomForestClassifier(max_depth=rf_max_depth,
                                            n_estimators=rf_n_estimators,
                                            min_samples_leaf=rf_min_samples_leaf,
                                            min_samples_split=rf_min_samples_split,
                                            bootstrap=rf_bootstrap,
                                            max_features=rf_max_features)

    # Step 3: Scoring method:

    score = model_selection.cross_val_score(classifier_obj, X_FE, y_train, scoring='f1')
    recall = score.mean()
    return recall


# Step 4: Running it
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)