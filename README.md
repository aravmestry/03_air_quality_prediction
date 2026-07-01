# Air Quality Prediction Studio

## Project Overview

This project predicts hourly PM2.5 air pollution concentrations using pollutant and meteorological variables from the Beijing Multi-Site Air Quality Dataset.

The goal is to build a reproducible environmental machine learning workflow that includes data cleaning, exploratory analysis, feature engineering, model comparison, cross-validation, and interactive visualization.

## Research Question

Can pollutant and meteorological variables be used to accurately predict PM2.5 concentrations across multiple air quality monitoring stations?

## Dataset

The dataset contains hourly air quality observations from 12 monitoring stations in Beijing from 2013 to 2017.

Target variable:

- PM2.5 concentration

Predictors include:

- PM10
- SO2
- NO2
- CO
- O3
- Temperature
- Pressure
- Dew point
- Rainfall
- Wind speed
- Season
- Weekend indicator
- Cyclical hour features

## Methods

Models compared:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest
- Extra Trees
- Gradient Boosting
- Hist Gradient Boosting

The best-performing model was Extra Trees Regression.

## Results

The Extra Trees model achieved approximately:

- R²: 0.946
- MAE: 11.1 µg/m³
- RMSE: 18.6 µg/m³

Five-fold cross-validation showed stable performance across folds.

## Dashboard

The Streamlit dashboard allows users to:

- Explore PM2.5 distributions
- Compare monitoring stations
- View model performance
- Adjust predictor values
- Estimate PM2.5 concentration interactively

## Technical Skills

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib
- seaborn
- Plotly
- Streamlit
- Machine learning regression
- Cross-validation
- Environmental data analysis