# Timeseries-Final-Project

link for the dataset https://www.kaggle.com/selfishgene/historical-hourly-weather-data.

The aim of the project is to forecast the temperature of San Francisco state. We will be plotting the correlation matrix to check the multicollinearity between the dependent variables. Then, we will perform ADF fuller test to check whether our dependent variable is stationary. Next, we will perform Time series decomposition and calculate the strength of trend and seasonality in the data. Then we will proceed with the modelling on our dataset starting with holt’s winter model followed by forward and backward step wise regression, ARMA, SARIMA and other base models like Average, Naïve, Drift, Simple Exponential Smoothing, Holt’s Linear method. We will estimate the parameters of the ARMA model using Levenberg Marquardt algorithm. Finally, we will be finding the best model for the dataset by comparing performance of various models using different metrics like mean square error, Q-value, root mean square error and perform one-step and h-step predictions on the best model.

Results:
Holt’s winter seasonal is better performing model compared to other models.
