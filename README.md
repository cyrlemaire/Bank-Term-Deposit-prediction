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
- Python 3.8 : Please find the documentation [at this address](https://www.python.org/downloads/)


## Getting Started

### 1. Clone this repository

```
$ git clone <this project>
$ cd <this project>
```

### 3. Setup your environment

First, please install the required dependencies with [Poetry](https://python-poetry.org).
In the folder containing the poetry.toml, write in your terminal
```
$ poetry install
```

### 4. setup config file

first you need to add the config file path to your environment variables. 

1. In config/ rename the file config_template.yml to config.yml
2. Create the file config/env.config
3. In the env.config file, write :
```
export CONFIG_PATH=<path of your yml file>
```
4. Write in your terminal :
```
source config/env.config
```
5. fill in the required path and file name in the sections "data" and "prediction_data" of the config.yml file. You files can be located in the data/ folder of the repository or in other folder, as long as the path is indicated in the config.yml file.


### 5 run the code and the jupyter notebooks

Can now run the scripts in the project from the subscription_forecast folder. For instance, to train thee model and obtain the precision/recall curve write:
```
 poetry run python application/train_model.py
```

If you want to launch the notebooks, write :
```
 poetry run python jupyter notebook
```

### 6. How to use the code

Once the config.yml is filled accordingly, you can just run the final_model_prediction.py script with:
```
 poetry run python application/final_model_prediction.py
```
This will train the model on the training data, display the precision/recall curve, and produce the predictions from the "prediction_data" datasets. The prediction vector will be stored in a csv file ("predictions_file_name" in thee config.yml). <br>
By default, the model is an optimized Random Forest with a threshold of 0.2. You can change that in the config.yml file.<br>
The hyperparmaters_optimisation.py script allows you to start an optimisation study for the model. Runninf this study might take about an hour or more.<br>
The 3 notebooks explain in more detail:
- The Exploratory Data Analysis
- The development of an optimal model
- The explanability of the model








