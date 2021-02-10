'''
Title: Helper Functions
Author: Akhil Kumar Baipaneni
Created on: 12/06/2020
Description: This file contains the helper functions required for the final project
'''

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from statsmodels.tsa.stattools import adfuller
from scipy.stats import chi2
from scipy import signal


#Average Method
def avg_method(train):
    y_hat_avg = np.mean(train)
    return y_hat_avg


# Naive Method
def naive_method(t):
    return t


# Drift method
def drift_method(t, h):
    y_hat_drift = t[len(t)-1] + h*((t[len(t)-1]-t[0])/(len(t) - 1))
    return y_hat_drift


#SES Method
def ses(t, damping_factor, l0):
    yhat4 = []
    yhat4.append(l0)
    for i in range(1, len(t)-1):
        res = damping_factor*(t[i]) + (1-damping_factor)*(yhat4[i-1])
        yhat4.append(res)
    return yhat4


def plot_func(var, label, x_label, y_label, title):
    plt.figure()
    plt.plot(var, label=label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.show()


def plot_acf(var_acf, lags, var_name):
    plt.figure()
    plt.stem(range(-(lags-1),lags), var_acf, use_line_collection=True)
    plt.xlabel('Lags')
    plt.ylabel('Magnitude')
    plt.title('Autocorrelation plot for {} variable'.format(var_name))
    plt.show()


def adf_cal(x):
    result = adfuller(x)
    print('ADF Statistic: %f' %result[0])
    print('p-value: %f' %result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))


def auto_corr(y, k):
    T = len(y)
    y_mean = np.mean(y)
    res_num = 0
    res_den = 0
    for t in range(k, T):
        res_num += (y[t] - y_mean) * (y[t-k] - y_mean)
    for t in range(0, T):
        res_den += (y[t] - y_mean)**2
    result = res_num/res_den
    return result


def cal_auto_corr(y, k):
    res = []
    res1 = []
    for t in range(0, k):
        result = auto_corr(y, t)
        res.append(result)
    for t in range(k-1, 0, -1):
        res1.append(res[t])
    res1.extend(res)
    return res1


def correlation_coefficent_cal(x, y):
    result = 0
    cov_res = 0
    var_res1 = 0
    var_res2 = 0
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    if len(x) == len(y):
        for i in range(0, len(x)):
            cov_res += ((x[i]-mean_x)*(y[i]-mean_y))
            var_res1 += (x[i]-mean_x)**2
            var_res2 += (y[i]-mean_y)**2
    result += cov_res/(np.sqrt(var_res1)*np.sqrt(var_res2))
    return result


def plot_ma(y, k, trend, detrend, ma_order, folding_order):
    plt.figure()
    plt.plot(np.array(y.index[:200]), np.array(y[:200]), label='original')
    if ma_order%2 != 0:
        plt.plot(np.array(y.index[k:200]), np.array(trend[:200-k]), label='{}-MA'.format(ma_order))
        plt.title('Plot for {}-MA'.format(ma_order))
    else:
        plt.plot(np.array(y.index[k:200]), np.array(trend[:200 - k]), label='{}x{}-MA'.format(folding_order, ma_order))
        plt.title('Plot for {}x{}-MA'.format(folding_order, ma_order))
    plt.plot(np.array(y.index[k:200]), np.array(detrend[:200-k]), label='detrended')
    plt.xlabel('DateTime')
    plt.ylabel('Temperature in Kelvin')
    plt.xticks(rotation=90)
    plt.legend()
    plt.show()


def cal_moving_average(y, ma_order, folding_order):
    ma = []
    k = int(np.ceil((ma_order - 1) / 2))
    for t in range(0, len(y) - ma_order + 1):
        temp = np.sum(y[t:ma_order + t])
        ma.append(temp / ma_order)

    if folding_order > len(ma):
        print("Invalid Folding order. Moving Average cannot be calculated if folding order is greater than the length of first moving average result")
    # passing folding order as zero for odd order of moving average
    elif folding_order != 0:
        k1 = int(np.ceil((ma_order - 1) / 2) + ((folding_order - 1) / 2))
        folding_ma = []
        for t in range(0, len(ma) - folding_order + 1):
            a = np.sum(y[t:folding_order + t])
            folding_ma.append(a / folding_order)
        print("Result of {}x{}-MA is: {}".format(folding_order, ma_order, folding_ma))
        detrended = np.subtract(list(y.iloc[k1:-k1]), folding_ma)
        plot_ma(y, k1, ma, detrended, ma_order, folding_order)
        return detrended, folding_ma
    else:
        print("Result of {}-MA is: {}".format(ma_order, ma))
        detrended = np.subtract(list(y.iloc[k:-k]), ma)
        plot_ma(y, k, ma, detrended, ma_order, folding_order)
        return detrended, ma

# Function to plot train, test and prediction values
def LR_plot_fun(train, test, y_hat_OLS, y_test_hat_OLS, title):
    year = pd.date_range(start='2012-10-01 13:00:00', end='2017-10-28 00:00:00', freq='1H')
    plt.figure(figsize=(10,8))
    # plt.plot(year[0:len(train)], train, label='Training set')
    # plt.plot(year[0:len(train)], y_hat_OLS, label='Prediction values')
    plt.plot(range(len(train), len(train)+len(test)), test, label='Testing set')
    plt.plot(range(len(train), len(train)+len(y_test_hat_OLS)), y_test_hat_OLS, label='forecast values')
    plt.xlabel('Time')
    plt.ylabel('Temperature in Kelvin')
    plt.title(title)
    plt.legend(loc='upper left')
    plt.show()


def cal_Q_value(residual_error, residual_error_acf, lags):
    k = len(residual_error)
    return k * np.sum(np.array(residual_error_acf[lags:]) ** 2)


def step0(na, nb):
    theta = np.zeros(shape=(na+nb,1))
    return theta.flatten()


def cal_wn(na, theta, y):
    num = [1] + list(theta[na:])
    den = [1] + list(theta[:na])
    if len(den) < len(num):
        den.extend([0 for i in range(len(num) - len(den))])
    elif len(num) < len(den):
        num.extend([0 for i in range(len(den) - len(num))])
    sys = (den, num, 1)
    _, e = signal.dlsim(sys, y)
    e = [item[0] for item in e]
    return np.array(e)


def step1(na, nb, theta, delta, y):
    X_i = []
    e = cal_wn(na, theta, y)
    sse_old = np.matmul(e.T, e)
    for i in range(na + nb):
        updated_theta = theta.copy()
        updated_theta[i] = theta[i] + delta
        e_new = cal_wn(na, updated_theta, y)
        x_i = (e - e_new) / delta
        X_i.append(x_i)
    X = np.column_stack(X_i)
    A = np.matmul(X.T, X)
    g = np.matmul(X.T, e)
    return sse_old, A, g


def step2(A, g, mu, na, nb, theta, y):
    mu_I = mu * np.identity(na + nb)
    delta_theta = np.matmul(np.linalg.inv(A + (mu_I)), g)
    theta_new = theta + delta_theta
    e_new = cal_wn(na, theta_new, y)
    sse_new = np.matmul(e_new.T, e_new)
    if np.isnan(sse_new):
        sse_new = 10 ** 10
    return delta_theta, theta_new, sse_new


def step3(y, na, nb):
    mu_factor = 10
    max_iterations = 100
    mu_max = 1e10
    delta = 1e-6
    mu = 0.01
    iterations = 0
    sse_list = []
    epsilon = 1e-3
    theta = step0(na, nb)
    while iterations < max_iterations:
        print("Iteration number: {}".format(iterations))
        sse_old, A, g = step1(na, nb, theta, delta, y)
        print("SSE old: {}".format(sse_old))
        if iterations == 0:
            sse_list.append(sse_old)
        delta_theta, theta_new, sse_new = step2(A, g, mu, na, nb, theta, y)
        print("SSE new: {}".format(sse_new))
        sse_list.append(sse_new)
        if sse_new < sse_old:
            if np.linalg.norm(delta_theta) < epsilon:
                theta_hat = theta_new
                var_error = sse_new/(len(y) - (na+nb))
                covariance_theta = var_error * np.linalg.inv(A)
                print("Algorithm converged")
                return theta_hat, var_error, covariance_theta, sse_list
            else:
                theta = theta_new
                mu /= mu_factor
        while sse_new >= sse_old:
            mu = mu * mu_factor
            if mu > mu_max:
                print("Error. mu has reached it's max value")
                return None, None, None, None
            delta_theta, theta_new, sse_new = step2(A, g, mu, na, nb, theta, y)
        theta = theta_new
        iterations += 1
        if iterations > max_iterations:
            print("Error. Reached maximum iterations")
            return None, None, None, None


def confidence_interval(covariance_theta, na, nb, theta_hat):
    print("Confidence Interval for Estimated parameters")
    for i in range(na):
        lower = theta_hat[i] - 2 * np.sqrt(covariance_theta[i][i])
        upper = theta_hat[i] + 2 * np.sqrt(covariance_theta[i][i])
        print('{} < a{} < {}'.format(lower, i+1, upper))

    for j in range(nb):
        lower = theta_hat[na+j] - 2 * np.sqrt(covariance_theta[na+j][na+j])
        upper = theta_hat[na+j] + 2 * np.sqrt(covariance_theta[na+j][na+j])
        print('{} < b{} < {}'.format(lower, j+1, upper))


def zeros_poles(theta_hat, na, nb):
    p = [1] + list(theta_hat[:na])
    z = [1] + list(theta_hat[na:])
    poles = np.roots(p)
    zeros = np.roots(z)
    print('Zeros : {}'.format(zeros))
    print('Poles : {}'.format(poles))
    return zeros, poles


def sse_plot(sse_list, title):
    plt.figure()
    plt.plot(sse_list, label = 'sum square error')
    plt.xlabel('# of Iterations')
    plt.ylabel('sum square error')
    plt.title(title + ': Sum Square Error')
    plt.legend()
    plt.show()


def chi_square_test(Q_value, lags, na, nb, alpha=0.01):
    DOF = lags - na - nb
    chi_critical = chi2.ppf(1-alpha, DOF)
    print("chi_critical: {}".format(chi_critical))
    if Q_value < chi_critical:
        print("The Residuals are White")
    else:
        print("Residuals are not White")


def phi_kk(ry, j, k):
    num = np.zeros((k, k), dtype=np.float64)
    den = np.zeros((k, k), dtype=np.float64)
    for b in range(k):
        for a in range(k):
            if b != k-1:
                num[a][b] = ry[np.abs(j + a - b)]
                den[a][b] = ry[np.abs(j + a - b)]
            else:
                num[a][b] = ry[np.abs(j + a + 1)]
                den[a][b] = ry[np.abs(j - k + a + 1)]
    numerator_det = np.round(np.linalg.det(num), 5)
    denominator_det = np.round(np.linalg.det(den), 5)
    phi = math.inf if denominator_det == 0.0 else np.round(np.divide(numerator_det, denominator_det), 3)
    return phi


def plot_gpac(df, T):
    sns.heatmap(df, annot=True)
    plt.xlabel('AR process (K)')
    plt.ylabel('MA process (j)')
    plt.title('Generalized Partial Autocorrelation(GPAC) ARMA {} samples'.format(T))
    plt.show()


def Cal_GPAC(ry, j, k):
    gpac_table = np.zeros(shape=(j, k), dtype=np.float64)
    for a in range(j):
        for b in range(1, k+1):
            gpac_table[a][b-1] = phi_kk(ry, a, b)
    return gpac_table