# Project N°1 - Advanced ML : Product Subscription - Yotta Academy

Copyright : Dan COHEN, Cyril LEMAIRE & Julien SADOUN

## Context

For this assignment, we had to create a model "from scratch" in order to predict whether a bank client will subscribe or not to a bank product based on a dataset given by the bank.


## Project Arborescence


    ├── README.md                      	<- The top-level README for developers using this project.
    │
    ├── data				<- Folder containing the data to use for training and predictions.
    │ 
    ├── poetry.lock                    	<- Lock file to secure the version of dependencies.
    │
    ├── pyproject.toml                 	<- Poetry file with dependencies.
    │
    └── subscription_forecast          	<- Source code for use in this project.
        ├── __init__.py                	<- Makes src a Python module.
        │
        ├── application                	<- Scripts to train model, optimize it, and  models and make predictions.
        │   ├── __init__.py
        │   ├── final_model_prediction.py
        │   ├── hyperparmaters_optimisation.py
        │   └── train_model.py
        │
        ├── config            		<- Scripts containing the settings.
        │   ├── __init__.py
        │   ├── config.py
        │   └── config_template.yml
        │
        ├── domain                     	<- Feature engineering pipeline, and functions to evaluate the model performance
        │   ├── __init__.py
        │   ├── feature_engineering.py
        │   └── model_evaluation.py
        │
        ├── infrastructure             	<- Scripts to load the raw data in a Pandas DataFrame ready for thee feature engineering pipeline.
        │   ├── __init__.py
        │   └── preprocessing.py
        │
        ├── interface            		<- Jupyter notebooks.

___
## Pre-requisites

There are the tools you need to install the software and how to install them :

- Poetry : Please find the documentation [at this address](https://python-poetry.org/docs/)
- Python 3.8.5 : Please find the documentation [at this address](https://www.python.org/downloads/release/python-385/)


## Getting Started

### 1. Clone this repository

```
$ git clone <this project>
$ cd <this project>
```

### 2. Setup your environment

First, please install the required dependencies with [Poetry](https://python-poetry.org).
In the folder containing the poetry.toml, write in your terminal
```
$ poetry install
```

### 3. Config file setup

First, you need to add the config file path to your environment variables. 

1. In config/ rename the file config_template.yml to config.yml
2. Create the file config/env.config
3. In the env.config file, write :
```
export CONFIG_PATH=<path of your yml file>
```
4. Write in your terminal :
```
source subscription_forecast/config/env.config
```
5. Fill in the required path and file names in the sections "data" and "prediction_data" of the config.yml file. Your files can be located in the data/ folder of the repository or in another folder, as long as the path is indicated in the config.yml file.


### 4. Run the code and the jupyter notebooks

You can now run the scripts in the project from the subscription_forecast folder. For instance, to train the model and obtain the precision/recall curve, you have to write:
```
 poetry run python application/train_model.py
```

If you want to launch the notebooks, you have to write :
```
 poetry run python jupyter notebook
```

### 5. How to use the code

Once the config.yml is filled accordingly, you can just run the final_model_prediction.py script with the command :
```
 poetry run python application/final_model_prediction.py
```
This will train the model on the training data, display the precision/recall curve, and produce the predictions from the "prediction_data" datasets. The prediction vector will be stored in a csv file ("predictions_file_name" in the config.yml). <br>
By default, the model is an optimized Random Forest with a threshold of 0.2. You can change that in the config.yml file.<br>
The hyperparmaters_optimisation.py script allows you to start an optimisation study for the model. Running this study might take about an hour or more.<br>
The 3 notebooks explain in more details:
- The Exploratory Data Analysis
- The development of an optimal model
- The explanability of the model








