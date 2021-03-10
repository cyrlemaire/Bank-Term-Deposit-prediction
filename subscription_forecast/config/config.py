import yaml


def read_yaml() -> dict:
    """Returns YAML file as dict,
    config_path must be filed in by user before everything else"""

    config_path = '/Users/cyrillemaire/Documents/Yotta/Project/productsubscription_dc_cl_js/' \
                  'subscription_forecast/config/config.yml'

    with open(config_path, 'r') as file_in:
        config = yaml.safe_load(file_in)
    return config

