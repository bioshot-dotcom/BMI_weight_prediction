# Predicting BMI from Health Indicators

This project predicts Body Mass Index (BMI) based on various health indicators using various machine learning models.

## Dataset

The dataset consists of two files:

1. X1.csv: This file contains all the feature variables. The variables include Age, Height, and several other categorical health indicators.
2. Y1.csv: This file contains the target variable - Weight.

## Installation

The following libraries are required to run this code:

- pandas
- matplotlib
- numpy
- sklearn
- scipy
- seaborn

## Usage

1. The data is first imported and merged into a single DataFrame. 
2. The merged DataFrame is preprocessed by converting categorical variables into numerical, calculating the BMI from height and weight, and removing anomalies and duplicates.
3. The processed data is then split into a training set and a testing set.
4. Features are selected based on their correlation with the target variable.
5. The data is then passed to various machine learning models - Multi-Layer Perceptron Regressor (MLPRegressor), Linear Regression (LinearRegressor), K-Nearest Neighbors (KNN), Support Vector Machine (SVM), and Random Forest Regressor.
6. Hyperparameter tuning is performed on each model using GridSearchCV.
7. Each model's performance is evaluated based on Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (r2) metrics.
8. The performances of the models are also visualized using bar plots.

## Project Files

The following are the main files included in this project:

- `data_preprocessing.ipynb`: Contains all the code for data preprocessing, model training, hyperparameter tuning, model evaluation, and visualization.
- `X1.csv`: Dataset file containing all the feature variables.
- `Y1.csv`: Dataset file containing the target variable.


## Results 








