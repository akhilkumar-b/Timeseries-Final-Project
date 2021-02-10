import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2
from sklearn.model_selection import train_test_split
from Final_Project.Toolbox import avg_method, naive_method, drift_method, ses, plot_func, plot_acf, cal_auto_corr, adf_cal, correlation_coefficent_cal, cal_moving_average, LR_plot_fun, cal_Q_value, Cal_GPAC, plot_gpac, step3, confidence_interval, zeros_poles, chi_square_test, sse_plot
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
import statsmodels.tsa.holtwinters as ets
import statsmodels.api as sm
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.stats import chi2
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse
import warnings
warnings.filterwarnings("ignore")

# The dataset contains ~5 years(October 2012- October 2017) of high temporal resolution (hourly measurements) data of various weather attributes,
# such as temperature, humidity, air pressure, etc of San Francisco State.
# Independent Variables - datetime, Humidity, Wind Speed, Wind Direction, Pressure, Weather Description
# datetime - hourly data
# Humidity - amount of water vapor in the air in percentage
# Wind Speed - the rate at which air is moving in a particular area (meters/second).
# Wind Direction - degree of direction
# Pressure - Standard sea level pressure in hectopascals
# Weather Description

# Dependent Variables - Temperature
# Temperature - San Francisco temperature in kelvin scale

# Reading the data
df = pd.read_csv('San Francisco Weather Data Oct2012-Oct2017.csv', index_col='datetime', parse_dates=True)

# Setting the frequency to hourly
df.index.freq = '1H'

# Let's see the first few rows in the dataset
pd.set_option('display.max_columns', None)
print(df.head())

# More information about the dataset
print(df.info())

# Replacing the Missing values with the previous values
df = df.fillna(method='ffill')
print(df.info())
df = df.drop(columns='Weather Description')
plot_func(df['Temperature'], 'temperature', 'time', 'temperature in Kelvin', 'Historical hourly Weather data 2012-2017')
lags = 50
sm.graphics.tsa.plot_acf(df['Temperature'], lags=lags, title='Autocorrelation of temperature(Original Dataset)')
plt.show()


def temp_pred(df, season):
    print("*************************************** {} Results *****************************************".format(season))
    temp = df['Temperature']
    print("Number of missing values in temperature variable: {}".format(temp.isna().sum()))
    temp_1 = temp.diff(periods=24)
    temp_1 = temp_1[24:]
    temp_2 = temp_1.diff()
    temp_2 = temp_2[1:]

    # dependent variable versus time
    plot_func(temp, 'temperature', 'time', 'temperature in Kelvin', 'Historical hourly Weather data 2012-2013')
    plot_func(temp_1, 'temperature', 'time', 'Magnitude', 'Historical hourly Weather data 2012-2013 (24th diff)')
    plot_func(temp_2, 'temperature', 'time', 'Magnitude', 'Historical hourly Weather data 2012-2013 (24th+1st diff)')

    # ACF of the dependent variable
    lags = 50
    sm.graphics.tsa.plot_acf(temp, lags=lags, title='Autocorrelation of temperature')
    plt.show()
    sm.graphics.tsa.plot_acf(temp_1, lags=lags, title='Autocorrelation of temperature(24th diff)')
    plt.show()
    sm.graphics.tsa.plot_acf(temp_2, lags=lags, title='Autocorrelation of temperature(24th+1st diff)')
    plt.show()
    sm.graphics.tsa.plot_pacf(temp, lags=lags, title='partial correlation of temperature')
    plt.show()
    sm.graphics.tsa.plot_pacf(temp_1, lags=lags, title='partial correlation of temperature(24th diff)')
    plt.show()
    sm.graphics.tsa.plot_pacf(temp_2, lags=lags, title='partial correlation of temperature(24th+1st diff)')
    plt.show()

    lags = 240
    acf_1 = sm.graphics.tsa.acf(temp_1, nlags=lags)
    plt.figure()
    plt.stem(range(0,lags+1)[::24], acf_1[::24], use_line_collection=True)
    plt.xlabel('Lags')
    plt.ylabel('Magnitude')
    plt.title('Autocorrelation plot for {} every 24 lags'.format('temperature(24th diff)'))
    plt.show()

    acf_2 = sm.graphics.tsa.acf(temp_2, nlags=lags)
    plt.figure()
    plt.stem(range(0, lags+1)[::24], acf_2[::24], use_line_collection=True)
    plt.xlabel('Lags')
    plt.ylabel('Magnitude')
    plt.title('Autocorrelation plot for {} every 24 lags'.format('temperature(24th+1st diff)'))
    plt.show()

    pacf_1 = sm.graphics.tsa.pacf(temp_1, nlags=lags)
    plt.figure()
    plt.stem(range(0, lags+1)[::24], pacf_1[::24], use_line_collection=True)
    plt.xlabel('Lags')
    plt.ylabel('Magnitude')
    plt.title('PartialAutocorrelation plot for {} every 24 lags'.format('temperature(24th diff)'))
    plt.show()

    pacf_2 = sm.graphics.tsa.pacf(temp_2, nlags=lags)
    plt.figure()
    plt.stem(range(0,lags+1)[::24], pacf_2[::24], use_line_collection=True)
    plt.xlabel('Lags')
    plt.ylabel('Magnitude')
    plt.title('PartialAutocorrelation plot for {} every 24 lags'.format('temperature(24th+1st diff)'))
    plt.show()

    # Correlation Matrix with seaborn heatmap and Pearson's correlation coefficent
    corrMatrix = df.corr()
    ax = sns.heatmap(corrMatrix, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200), square=True, annot=True)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.show()

    r_ht = correlation_coefficent_cal(df['Humidity'], df.Temperature)
    print("The correlation coefficient between the Humidity and Temperature is {:.3f}".format(r_ht))
    r_wst = correlation_coefficent_cal(df['Wind Speed'], df.Temperature)
    print("The correlation coefficient between the Wind Speed and Temperature is {:.3f}".format(r_wst))
    r_wdt = correlation_coefficent_cal(df['Wind Direction'], df.Temperature)
    print("The correlation coefficient between the Wind Direction and Temperature is  {:.3f}".format(r_wdt))
    r_pt = correlation_coefficent_cal(df['Pressure'], df.Temperature)
    print("The correlation coefficient between the Pressure and Temperature is  {:.3f}".format(r_pt))

    df['Temperature'].plot.hist(bins=20, grid=True, edgecolor='k').autoscale(enable=True, axis='both', tight=True)
    plt.xlabel('Temperature in Kelvin')
    plt.ylabel('Frequency')
    plt.title('Histogram plot of Temperature distribution')
    plt.show()

    adf_cal(temp)
    adf_cal(temp_1)
    adf_cal(temp_2)

    # Detrending the data using Moving Average method
    detrended_2x4, ma_2x4 = cal_moving_average(temp, ma_order=4, folding_order=2)
    adf_cal(detrended_2x4)

    # Time series decomposition using STL(Seasonal and Trend decomposition using Loess) method
    STL1 = STL(temp)
    res = STL1.fit()
    fig = res.plot()
    plt.show()

    T = res.trend
    S = res.seasonal
    R = res.resid

    plt.figure()
    plt.plot(T, label='trend')
    plt.plot(S, label='Seasonal')
    plt.plot(R, label='residuals')
    plt.xlabel('Year')
    plt.ylabel('Magnitude')
    plt.title('Trend, Seasonality, Residual components using STL Decomposition')
    plt.legend()
    plt.show()

    detrended = temp-T
    plt.figure()
    plt.plot(temp, label='Original')
    plt.plot(detrended, label='detrended')
    plt.xlabel('Year')
    plt.ylabel('Magnitude')
    plt.title('Original vs detrended')
    plt.legend()
    plt.show()

    adjusted_seasonal = temp-S
    plt.figure()
    plt.plot(temp, label='Original')
    plt.plot(adjusted_seasonal, label='Seasonally Adjusted')
    plt.xlabel('Year')
    plt.ylabel('Magnitude')
    plt.title('Original vs Seasonally adjusted')
    plt.legend()
    plt.show()

    # Measuring strength of trend and seasonality
    F = np.max([0,1-np.var(np.array(R))/np.var(np.array(T+R))])
    print('Strength of trend for Hourly weather dataset is {:.3f}'.format(F))

    FS = np.max([0, 1-np.var(np.array(R))/np.var(np.array(S+R))])
    print('Strength of seasonality for Hourly weather dataset is {:.3f}'.format(FS))


    # Average, Naive, Drift, Simple Exponential Smoothing, Holt's Linear and Holt's winter Seasonal Methods
    print("--------------Average, Naive, Drift, Simple Exponential Smoothing, Holt's Linear and Holt's winter Seasonal Methods-----------------")
    train, test = train_test_split(temp, shuffle=False, test_size=0.2)
    train.index.freq = '1H'
    test.index.freq = '1H'
    h = len(test)

    train_pred_avg = []
    for i in range(1,len(train)):
        res = avg_method(train.iloc[0:i])
        train_pred_avg.append(res)

    test_forecast_avg1 = np.ones(len(test)) * avg_method(train)
    test_forecast_avg = pd.DataFrame(test_forecast_avg1).set_index(test.index)
    residual_error_avg = np.array(train[1:]) - np.array(train_pred_avg)
    forecast_error_avg = test - test_forecast_avg1
    MSE_train_avg = np.mean((residual_error_avg)**2)
    MSE_test_avg = np.mean((forecast_error_avg)**2)
    mean_pred_avg = np.mean(residual_error_avg)
    mean_forecast_avg = np.mean(forecast_error_avg)
    print('Mean of prediction errors for Average method: ', mean_pred_avg)
    print('Mean of forecast errors for Average method: ', mean_forecast_avg)
    var_pred_avg = np.var(residual_error_avg)
    var_forecast_avg = np.var(forecast_error_avg)

    naive_train_pred = []
    for i in range(0, len(train)-1):
        res = naive_method(train[i])
        naive_train_pred.append(res)

    res = np.ones(len(test)) * train[-1]
    naive_test_forecast1 = np.ones(len(test)) * res
    naive_test_forecast = pd.DataFrame(naive_test_forecast1).set_index(test.index)
    residual_error_naive = np.array(train[1:]) - np.array(naive_train_pred)
    forecast_error_naive = test - naive_test_forecast1
    MSE_train_naive = np.mean((residual_error_naive)**2)
    MSE_test_naive = np.mean((forecast_error_naive)**2)
    mean_pred_naive = np.mean(residual_error_naive)
    mean_forecast_naive = np.mean(forecast_error_naive)
    print('Mean of prediction errors for Naive method: ', mean_pred_naive)
    print('Mean of forecast errors for Naive method: ', mean_forecast_naive)
    var_pred_naive = np.var(residual_error_naive)
    var_forecast_naive = np.var(forecast_error_naive)

    drift_train_forecast=[]
    for i in range(1, len(train)):
        if i == 1:
            drift_train_forecast.append(train[0])
        else:
            h = 1
            res = drift_method(train[0:i], h)
            drift_train_forecast.append(res)

    drift_test_forecast1=[]
    for h in range(1, len(test)+1):
        res = drift_method(train, h)
        drift_test_forecast1.append(res)

    drift_test_forecast = pd.DataFrame(drift_test_forecast1).set_index(test.index)
    residual_error_drift = np.array(train[1:]) - np.array(drift_train_forecast)
    forecast_error_drift = np.array(test) - np.array(drift_test_forecast1)
    MSE_train_drift = np.mean((residual_error_drift)**2)
    MSE_test_drift = np.mean((forecast_error_drift)**2)
    mean_pred_drift = np.mean(residual_error_drift)
    mean_forecast_drift = np.mean(forecast_error_drift)
    print('Mean of prediction errors for Drift method: ', mean_pred_drift)
    print('Mean of forecast errors for Drift method: ', mean_forecast_drift)
    var_pred_drift = np.var(residual_error_drift)
    var_forecast_drift = np.var(forecast_error_drift)

    l0 = train[0]
    ses_train_pred = ses(train, 0.50, l0)
    ses_test_forecast1 = np.ones(len(test)) * (0.5*(train[-1]) + (1-0.5)*(ses_train_pred[-1]))
    ses_test_forecast = pd.DataFrame(ses_test_forecast1).set_index(test.index)
    residual_error_ses = np.array(train[1:]) - np.array(ses_train_pred)
    forecast_error_ses = np.array(test) - np.array(ses_test_forecast1)
    MSE_train_SES = np.mean((residual_error_ses)**2)
    MSE_test_SES = np.mean((forecast_error_ses)**2)
    mean_pred_SES = np.mean(residual_error_ses)
    mean_forecast_SES = np.mean(forecast_error_ses)
    print('Mean of prediction errors for SES method: ', mean_pred_SES)
    print('Mean of forecast errors for SES method: ', mean_forecast_SES)
    var_pred_SES = np.var(residual_error_ses)
    var_forecast_SES = np.var(forecast_error_ses)

    # SES Method using statsmodels for alpha=0.5
    # ses_train = train.ewm(alpha=0.5, adjust=False).mean()  # Another way of doing it
    ses_model1 = SimpleExpSmoothing(train)
    ses_fitted_model1 = ses_model1.fit(smoothing_level=0.5, optimized=False)
    ses_train_pred1 = ses_fitted_model1.fittedvalues.shift(-1)
    ses_test_forecast1 = ses_fitted_model1.forecast(steps=len(test))
    ses_test_forecast1 = pd.DataFrame(ses_test_forecast1).set_index(test.index)
    MSE_test_SES1 = np.square(np.subtract(test.values, np.ndarray.flatten(ses_test_forecast1.values))).mean()

    # Holt's Linear Trend
    holtl_fitted_model = ets.ExponentialSmoothing(train, trend='additive', damped=True, seasonal=None).fit()
    holtl_train_pred = holtl_fitted_model.fittedvalues
    holtl_test_forecast = holtl_fitted_model.forecast(steps=len(test))
    holtl_test_forecast = pd.DataFrame(holtl_test_forecast).set_index(test.index)
    residual_error_holtl = np.subtract(train.values, np.ndarray.flatten(holtl_train_pred.values))
    forecast_error_holtl = np.subtract(test.values, np.ndarray.flatten(holtl_test_forecast.values))
    MSE_train_holtl = np.mean((residual_error_holtl)**2)
    MSE_test_holtl = np.mean((forecast_error_holtl)**2)
    mean_pred_holtl = np.mean(residual_error_holtl)
    mean_forecast_holtl = np.mean(forecast_error_holtl)
    print("Mean of prediction errors for Holt's Linear method: ", mean_pred_holtl)
    print("Mean of forecast errors for Holt's Linear method: ", mean_forecast_holtl)
    var_pred_holtl = np.var(residual_error_holtl)
    var_forecast_holtl = np.var(forecast_error_holtl)

    # Holt's Winter Seasonal Trend
    holtw_fitted_model = ets.ExponentialSmoothing(train, trend='add', damped=True, seasonal='mul', seasonal_periods=24).fit()
    holtw_train_pred = holtw_fitted_model.fittedvalues
    holtw_test_forecast = holtw_fitted_model.forecast(steps=len(test))
    holtw_test_forecast = pd.DataFrame(holtw_test_forecast).set_index(test.index)
    residual_error_holtw = np.subtract(train.values, np.ndarray.flatten(holtw_train_pred.values))
    forecast_error_holtw = np.subtract(test.values, np.ndarray.flatten(holtw_test_forecast.values))
    MSE_train_holtw = np.mean((residual_error_holtw)**2)
    MSE_test_holtw = np.mean((forecast_error_holtw)**2)
    mean_pred_holtw = np.mean(residual_error_holtw)
    mean_forecast_holtw = np.mean(forecast_error_holtw)
    print("Mean of prediction errors for Holt's Winter Seasonal method: ", mean_pred_holtw)
    print("Mean of forecast errors for Holt's Winter Seasonal method: ", mean_forecast_holtw)
    var_pred_holtw = np.var(residual_error_holtw)
    var_forecast_holtw = np.var(forecast_error_holtw)

    fig, ax = plt.subplots(figsize=(10,8))
    ax.plot(train, label='Training set')
    ax.plot(test, label='Testing set')
    ax.plot(test_forecast_avg, label='Average h-step prediction')
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.title('Average Method')
    plt.legend(loc='upper left')
    plt.show()

    fig, ax = plt.subplots(figsize=(10,8))
    ax.plot(train, label='Training set')
    ax.plot(test, label='Testing set')
    ax.plot(naive_test_forecast, label='Naive h-step prediction')
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.title('Naive Method')
    plt.legend(loc='upper left')
    plt.show()

    fig, ax = plt.subplots(figsize=(10,8))
    ax.plot(train, label='Training set')
    ax.plot(test, label='Testing set')
    ax.plot(drift_test_forecast, label='Drift h-step prediction')
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.title('Drift Method')
    plt.legend(loc='upper left')
    plt.show()

    fig, ax = plt.subplots(figsize=(10,8))
    ax.plot(train, label='Training set')
    ax.plot(test, label='Testing set')
    ax.plot(ses_test_forecast, label='Simple Exponential Smoothing h-step prediction')
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.title('SES Method')
    plt.legend(loc='upper left')
    plt.show()

    fig, ax = plt.subplots(figsize=(10,8))
    ax.plot(train, label='Training set')
    ax.plot(test, label='Testing set')
    ax.plot(holtl_test_forecast, label="Holt's Linear h-step prediction")
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.title("Holt's Linear Method")
    plt.legend(loc='upper left')
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(train, label='Training set')
    ax.plot(test, label='Testing set')
    ax.plot(holtw_test_forecast, label="Holt's Winter Seasonal h-step prediction")
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.title("Holt's Winter Seasonal Method")
    plt.legend(loc='upper left')
    plt.show()

    fig, ax = plt.subplots(figsize=(10,8))
    # ax.plot(train, label='Training set')
    ax.plot(test, label='Testing set')
    ax.plot(holtw_test_forecast, label="Holt's Winter Seasonal h-step prediction")
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.title("Holt's Winter Seasonal Method")
    plt.legend(loc='upper left')
    plt.show()

    # Auto_correlation for Residual errors and Q value for Residual errors
    #Average Method
    k = len(test)
    lags = 30
    avg_residual_acf = cal_auto_corr(residual_error_avg, lags)
    Q_residual_avg = k * np.sum(np.array(avg_residual_acf[lags:])**2)
    plt.figure()
    plt.stem(range(-(lags-1),lags), avg_residual_acf, use_line_collection=True)
    plt.xlabel('Lags')
    plt.ylabel('Magnitude')
    plt.title('Autocorrelation plot for Residual Error (Average Method)')
    plt.show()

    # Naive method
    k = len(test)
    lags = 30
    naive_residual_acf = cal_auto_corr(residual_error_naive, lags)
    Q_residual_naive = k * np.sum(np.array(naive_residual_acf[lags:])**2)
    plt.figure()
    plt.stem(range(-(lags-1),lags), naive_residual_acf, use_line_collection=True)
    plt.xlabel('Lags')
    plt.ylabel('Magnitude')
    plt.title('Autocorrelation plot for Residual Error (Naive Method)')
    plt.show()

    # Drift Method
    k = len(test)
    lags = 30
    drift_residual_acf = cal_auto_corr(residual_error_drift, lags)
    Q_residual_drift = k * np.sum(np.array(drift_residual_acf[lags:])**2)
    plt.figure()
    plt.stem(range(-(lags-1),lags), drift_residual_acf, use_line_collection=True)
    plt.xlabel('Lags')
    plt.ylabel('Magnitude')
    plt.title('Autocorrelation plot for Residual Error (Drift Method)')
    plt.show()

    # SES method
    k = len(test)
    lags = 30
    ses_residual_acf = cal_auto_corr(residual_error_ses, lags)
    Q_residual_SES = k * np.sum(np.array(ses_residual_acf[lags:])**2)
    plt.figure()
    plt.stem(range(-(lags-1), lags), ses_residual_acf, use_line_collection=True)
    plt.xlabel('Lags')
    plt.ylabel('Magnitude')
    plt.title('Autocorrelation plot for Residual Error (SES Method)')
    plt.show()

    # holt's linear method
    k = len(test)
    lags = 30
    holtl_residual_acf = cal_auto_corr(residual_error_holtl, lags)
    Q_residual_holtl = k * np.sum(np.array(holtl_residual_acf[lags:])**2)
    plt.figure()
    plt.stem(range(-(lags-1), lags), holtl_residual_acf, use_line_collection=True)
    plt.xlabel('Lags')
    plt.ylabel('Magnitude')
    plt.title("Autocorrelation plot for Residual Error (Holt's Linear Method)")
    plt.show()

    # holt's Winter Seasonal method
    k = len(train)
    lags = 30
    holtw_residual_acf = cal_auto_corr(residual_error_holtw, lags)
    Q_residual_holtw = k * np.sum(np.array(holtw_residual_acf[lags:])**2)
    print("Q-value of Residual error for Holts winter method: {}".format(Q_residual_holtw))
    plt.figure()
    plt.stem(range(-(lags-1), lags), holtw_residual_acf, use_line_collection=True)
    plt.xlabel('Lags')
    plt.ylabel('Magnitude')
    plt.title("Autocorrelation plot for Residual Error (Holt's winter Seasonal Method)")
    plt.show()

    sm.graphics.tsa.plot_acf(holtw_residual_acf, lags=lags, title="Autocorrelation for Residual Error (Holt's winter Seasonal Method)")
    plt.show()

    corr_avg = correlation_coefficent_cal(forecast_error_avg, test)
    corr_naive = correlation_coefficent_cal(forecast_error_naive, test)
    corr_drift = correlation_coefficent_cal(forecast_error_drift, test)
    corr_ses = correlation_coefficent_cal(forecast_error_ses, test)
    corr_holtl = correlation_coefficent_cal(forecast_error_holtl, test)
    corr_holtw = correlation_coefficent_cal(forecast_error_holtw, test)
    d = {'Methods':['Average', 'Naive', 'Drift', 'SES', "HoltL", "HoltW"],
         'Q_val': [round(Q_residual_avg, 2), round(Q_residual_naive, 2), round(Q_residual_drift,2), round(Q_residual_SES,2), round(Q_residual_holtl,2), round(Q_residual_holtw,2)],
         'MSE(P)': [round(MSE_train_avg,2), round(MSE_train_naive,2), round(MSE_train_drift,2), round(MSE_train_SES,2), round(MSE_train_holtl,2), round(MSE_train_holtw,2)],
         'MSE(F)': [round(MSE_test_avg,2), round(MSE_test_naive,2), round(MSE_test_drift,2), round(MSE_test_SES,2), round(MSE_test_holtl,2), round(MSE_test_holtw,2)],
         'var(P)': [round(var_pred_avg,2), round(var_pred_naive,2), round(var_pred_drift,2), round(var_pred_SES,2), round(var_pred_holtl,2), round(var_pred_holtw,2)],
         'var(F)':[round(var_forecast_avg,2), round(var_forecast_naive,2), round(var_forecast_drift,2), round(var_forecast_SES,2), round(var_forecast_holtl,2), round(var_forecast_holtw,2)],
         'corrcoeff':[round(corr_avg,2), round(corr_naive,2), round(corr_drift,2), round(corr_ses,2), round(corr_holtl,2), round(corr_holtw,2)]}
    df1 = pd.DataFrame(data=d)
    df1 = df1.set_index('Methods')
    pd.set_option('display.max_columns', None)
    print(df1)


    # Forward step regression
    df2 = df[['Humidity', 'Temperature']]
    features = df2.drop(columns='Temperature')
    target = df2['Temperature']
    features = sm.add_constant(features)
    x_train, x_test, y_train, y_test = train_test_split(features, target, shuffle=False, test_size=0.2)
    model = sm.OLS(y_train, x_train).fit()
    print(model.summary())

    df2 = df[['Humidity', 'Wind Speed', 'Temperature']]
    features = df2.drop(columns='Temperature')
    target = df2['Temperature']
    features = sm.add_constant(features)
    x_train, x_test, y_train, y_test = train_test_split(features, target, shuffle=False, test_size=0.2)
    model = sm.OLS(y_train, x_train).fit()
    print(model.summary())

    df2 = df[['Humidity', 'Wind Speed', 'Wind Direction', 'Temperature']]
    features = df2.drop(columns='Temperature')
    target = df2['Temperature']
    features = sm.add_constant(features)
    x_train, x_test, y_train, y_test = train_test_split(features, target, shuffle=False, test_size=0.2)
    model = sm.OLS(y_train, x_train).fit()
    print(model.summary())

    df2 = df[['Humidity', 'Wind Speed', 'Wind Direction', 'Pressure', 'Temperature']]
    features = df2.drop(columns='Temperature')
    target = df2['Temperature']
    features = sm.add_constant(features)
    x_train, x_test, y_train, y_test = train_test_split(features, target, shuffle=False, test_size=0.2)
    model = sm.OLS(y_train, x_train).fit()
    print(model.summary())

    df2 = df[['Humidity', 'Wind Speed', 'Wind Direction', 'Temperature']]
    features = df2.drop(columns='Temperature')
    target = df2['Temperature']
    features = sm.add_constant(features)
    x_train, x_test, y_train, y_test = train_test_split(features, target, shuffle=False, test_size=0.2)
    model = sm.OLS(y_train, x_train).fit()
    print(model.summary())

    # Backward step regression
    df2 = df[['Humidity', 'Wind Speed', 'Wind Direction', 'Pressure', 'Temperature']]
    features = df2.drop(columns='Temperature')
    target = df2['Temperature']
    features = sm.add_constant(features)
    x_train, x_test, y_train, y_test = train_test_split(features, target, shuffle=False, test_size=0.2)
    model = sm.OLS(y_train, x_train).fit()
    print(model.summary())

    df2 = df[['Humidity', 'Wind Speed', 'Wind Direction', 'Temperature']]
    features = df2.drop(columns='Temperature')
    target = df2['Temperature']
    features = sm.add_constant(features)
    x_train, x_test, y_train, y_test = train_test_split(features, target, shuffle=False, test_size=0.2)
    model = sm.OLS(y_train, x_train).fit()
    print(model.summary())

    # 1-step ahead prediction
    y_hat_OLS = model.predict(x_train)
    y_test_hat_OLS = model.predict(x_test)
    LR_plot_fun(y_train, y_test, y_hat_OLS, y_test_hat_OLS, 'OLS Regression Method')

    prediction_error = y_train - y_hat_OLS
    forecast_error = y_test - y_test_hat_OLS
    lags = 30
    prediction_error_acf = cal_auto_corr(prediction_error, lags)
    forecast_error_acf = cal_auto_corr(forecast_error, lags)
    plot_acf(prediction_error_acf, lags=lags, var_name='OLS prediction error')
    plot_acf(forecast_error_acf, lags=lags, var_name='OLS forecast error')
    Q_value = cal_Q_value(prediction_error, prediction_error_acf, lags)

    T = len(x_train)
    K = len(x_train.columns)
    pred_var = (1/(T-K-1)) * (np.sum((prediction_error)**2))
    pred_std = np.sqrt((1/(T-K-1)) * (np.sum((prediction_error)**2)))
    print("Q value of the residual error: {:.2f}".format(Q_value))
    print("mean of prediction error: {:.2f}".format(np.mean(prediction_error)))
    print("variance of prediction error: ", pred_var)
    print("standard deviation of prediction error: ", pred_std)
    print("RMSE of prediction error: ", np.sqrt(np.mean(prediction_error**2)))

    T = len(x_test)
    K = len(x_test.columns)
    forecast_var = (1/(T-K-1)) * (np.sum((forecast_error)**2))
    forecast_std = np.sqrt((1/(T-K-1)) * (np.sum((forecast_error)**2)))
    print("mean of forecast error: {:.2f}".format(np.mean(forecast_error)))
    print("variance of forecast error: ", forecast_var)
    print("standard deviation of forecast error: ", forecast_std)
    print("RMSE of forecast error: ", np.sqrt(np.mean(forecast_error**2)))

    corr_coeff = round(correlation_coefficent_cal(y_test, y_test_hat_OLS),2)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_test_hat_OLS, c='green', alpha=1, label='y_test vs y_test_hat_OLS')
    plt.xlabel('y_test')
    plt.ylabel('y_test_hat_OLS')
    plt.title("Scatter plot of y_test vs y_hat_test with correlation coefficient of {}".format(corr_coeff))
    plt.legend()
    plt.show()

    corr_coeff1 = round(correlation_coefficent_cal(y_train, y_hat_OLS),2)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_train, y_hat_OLS, c='green', alpha=1, label='y_test vs y_test_hat_OLS')
    plt.xlabel('y_train')
    plt.ylabel('y_hat_OLS')
    plt.title("Scatter plot of y_train vs y_hat_OLS with correlation coefficient of {}".format(corr_coeff1))
    plt.legend()
    plt.show()
    return temp, temp_1, temp_2


winter_data = df[1453:3613]
spring_data = df[3613:5821]
summer_data = df[5821:8029]
fall_data = df[8029:10213]

df = summer_data
temp, temp_1, temp_2 = temp_pred(df, "summer")

# GPAC
y = temp_2.copy()
y_train, y_test = train_test_split(y, shuffle=False, test_size=0.2)
j=8; k=8
lags = 30
T = len(y_train)
ry = cal_auto_corr(y_train, lags)
plot_acf(ry, lags=lags, var_name='temperature(2nd Diff)')
gpac_table = Cal_GPAC(ry[lags-1:], j=8, k=8)
data_frame = pd.DataFrame(gpac_table)
data_frame.columns = np.arange(1, k+1)
plot_gpac(data_frame, T=T)

print("potential order of AR and MA is 2 and 1 respectively")

# LM Algorithm
na = 2
nb = 1

# ARMA parameter Estimation
theta_hat, var_error, covariance_theta, sse_list = step3(y_train, na, nb)
an = []
bn = []
for i in range(na):
    an.append(theta_hat[i])
    print('The AR coefficient a{}'.format(i), 'is: ', theta_hat[i])
for i in range(nb):
    bn.append(theta_hat[i+na])
    print('The MA coefficient a{}'.format(i), 'is: ', theta_hat[i+na])

print("Final Parameters are: {}".format(theta_hat))
print("Standard deviation of the parameter estimates:{}".format(covariance_theta.diagonal()))
print("Covariance Matrix: {}".format(covariance_theta))
print("estimated variance of error: {}".format(var_error))
confidence_interval(covariance_theta, na, nb, theta_hat)
zeros, poles = zeros_poles(theta_hat, na, nb)
sse_plot(sse_list, title='y(t) - 0.83y(t - 1) -0.08y(t-1) = e(t) -0.99 (t-1) SSE Plot')

# 1- step prediction
y_hat_t_1 = []
for i in range(0, len(y_train)):
    if i == 0:
        y_hat_t_1.append(-theta_hat[0] * y_train[i] + theta_hat[2]*y_train[i])
    else:
        y_hat_t_1.append(-theta_hat[0] * y_train[i] - theta_hat[1] * y_train[i-1] + theta_hat[2]*(y_train[i] - y_hat_t_1[i - 1]))

x = [i for i in range(50)]
plt.figure()
plt.plot(x[1:], y_train[1:50], label="Training Set")
plt.plot(x[1:], y_hat_t_1[:49], label="One-step Prediction")
plt.xlabel("TimeSteps")
plt.ylabel("Y Values")
plt.legend()
plt.title("One Step Prediction of ARMA model")
plt.show()

residual_errors = np.subtract(y_train[1:], y_hat_t_1[:-1])
print('mean of residual errors: ', np.mean(residual_errors))
print('variance of residual error : ', np.var(residual_errors))
k = len(y_train)-1
residual_error_acf = cal_auto_corr(residual_errors, lags)
Q_value = k * np.sum(np.array(residual_error_acf[lags:])**2)
print("Q-value: {}".format(Q_value))
chi_square_test(Q_value, lags, na, nb)

# h-step ahead prediction
y_hat_t_h = []
for h in range(len(y_test)):
    if h == 0:
        y_hat_t_h.append(-theta_hat[0] * y_train[h-1] - theta_hat[1] * y_train[h-2]+ theta_hat[2]*(y_train[h-1] - y_hat_t_1[h-2]))
    elif h==1:
        y_hat_t_h.append(-theta_hat[0] * y_hat_t_h[h-1] - theta_hat[1] * y_train[h-2])
    else:
        y_hat_t_h.append(-theta_hat[0] * y_hat_t_h[h - 1] - theta_hat[1] * y_hat_t_h[h-2])

forecast_error = np.subtract(y_test, y_hat_t_h)
print('Variance of forecast error : ', np.var(forecast_error))

plt.figure()
plt.plot(x, y_test[:50], label='Test set')
plt.plot(x, y_hat_t_h[:50], label='h-step ahead prediction')
plt.xlabel("TimeSteps")
plt.ylabel("Y Values")
plt.title('h-step ahead prediction of ARMA model')
plt.legend()
plt.show()

# SARIMA Model

train, test = train_test_split(temp, shuffle=False, test_size=0.2)
train.index.freq = '1H'
test.index.freq = '1H'
model = SARIMAX(train, order=(2,1,1), seasonal_order=(0,1,1,24))
results = model.fit()
print(results.summary())
# one-step prediction
one_step = results.fittedvalues[1:]
sarima_residual_errors = np.subtract(train[1:], one_step)
sarima_residual_mse = mean_squared_error(train[1:], one_step)
print('SARIMA(2,1,1)(0,1,1,24) MSE Residual Error: {}'.format(sarima_residual_mse))
sarima_mean_residual_error = np.mean(sarima_residual_errors)
sarima_var_residual_error = np.mean(sarima_residual_errors)
print('variance of residual error : ', np.var(sarima_residual_errors))

# Plot of one-step prediction against train values
title = 'Train data vs one-step prediction - SARIMA model'
ylabel='Temperature'
xlabel='Time'
plt.figure()
plt.plot(train[31:], label='train data')
plt.plot(one_step[30:], label='one-step ahead prediction')
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title)
plt.legend()
plt.show()

def chi_square_test1(Q_value, lags, na, nb, Na, Nb,alpha=0.01):
    DOF = lags - na - nb-Na-Nb
    chi_critical = chi2.ppf(1-alpha, DOF)
    print("chi_critical: {}".format(chi_critical))
    if Q_value < chi_critical:
        print("The Residuals are White")
    else:
        print("Residuals are not White")

k = len(train)-1
lags = 50
na = 1
nb = 1
Na = 0
Nb = 1
residual_error_acf = cal_auto_corr(sarima_residual_errors, lags)
Q_value = k * np.sum(np.array(residual_error_acf[lags:])**2)
print("Q-value: {}".format(Q_value))
chi_square_test1(Q_value, lags, na, nb, Na, Nb)
plt.figure()
plt.stem(range(-(lags - 1), lags), residual_error_acf, use_line_collection=True)
plt.xlabel('Lags')
plt.ylabel('Magnitude')
plt.title('Autocorrelation plot for Residual Error SARIMA model')
plt.show()

# Obtain predicted values
start=len(train)
end=len(train)+len(test[:200])-1
predictions = results.predict(start=start, end=end, dynamic=False, typ='levels').rename('SARIMA(2,1,1)(0,1,1,24) Predictions')
sarima_forecast_errors = np.subtract(test, predictions)
sarima_mean_forecast_error = np.mean(sarima_forecast_errors)
sarima_var_forecast_error = np.var(sarima_forecast_errors)
print('variance of forecast error : ', np.var(sarima_forecast_errors))

# h-step predictions against known values
title = 'Test Data vs h-step prediction- SARIMA model'
ylabel='Temperature'
xlabel='Time'
plt.figure()
plt.plot(test[:200], label='test data')
plt.plot(predictions, label='h-step prediction')
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title)
plt.legend()
plt.show()

sarima_forecast_mse = mean_squared_error(test[:200], predictions)
print('SARIMA(2,1,1)(0,1,1,24) MSE forecast Error: {}'.format(sarima_forecast_mse))
sarima_rmse = rmse(test[:200], predictions)
print('SARIMA(2,1,1)(0,1,1,24) RMSE forecast Error: {}'.format(sarima_rmse))
corr_sarima = correlation_coefficent_cal(test[:200], predictions)