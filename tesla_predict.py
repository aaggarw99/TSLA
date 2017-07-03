# source: https://finance.yahoo.com/quote/TSLA/history?period1=1467139781&period2=1498675781&interval=1d&filter=history&frequency=1d

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_boston
import scipy as sp



dates = []
prices = []

def get_data(filename):
    global dates, prices
    df = pd.read_csv(filename)
    dates = df.index.values
    prices = df["Open"].values

def plot_data(d, p, price_type):
    plt.plot(d, p)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Tesla "+price_type+" Price vs Date")
    plt.ylim(min(p)-5, max(p)+5)
    plt.xlim(min(d)-5, max(d)+5)
    plt.show()


def train_data(dates, prices):
    dates = [[date] for date in dates]
    # prices = [[price] for price in prices]
    # print(dates, prices)

    # dates = np.reshape(dates, (len(dates), 1))
    # print(dates, prices)
    X_train = dates[:-100]
    X_test = dates[-100:]

    Y_train = prices[:-100]
    # compare results to this
    Y_test = prices[-100:]
    plt.scatter(dates, prices, color="black", label="Data")

    # regr = linear_model.LinearRegression()
    # regr.fit(X_train, Y_train)
    #
    # print(regr.coef_)
    #
    # return regr.score(X_test, Y_test)

    svr_poly = SVR(kernel="poly", C=1e3, cache_size=7000, degree=2)
    # svr_poly.fit(X_train, Y_train)

    svr_rbf = SVR(kernel="rbf", C=1e3, cache_size=7000, gamma = 0.1)
    svr_rbf.fit(X_train, Y_train)

    svr_lin = SVR(kernel="linear", C=1e3, cache_size=7000)
    svr_lin.fit(X_train, Y_train)

    plt.plot(X_test, svr_rbf.predict(X_test), color="green", label="RBF")
    plt.plot(X_test, svr_lin.predict(X_test), color="red", label="Linear")
    # plt.plot(X_test, svr_poly.predict(X_test), color="blue", label="Poly")
    plt.legend()
    plt.show()
    # add weights to things closer dates

    return

# trains model on alternating data points
def train_data_alt(dates, prices):
    cut = -100
    dates = [[date] for date in dates]
    X_train_raw = dates[:cut]
    X_test = dates[cut:]

    Y_train_raw = prices[:cut]
    Y_test = prices[cut:]

    X_train_alt = []
    Y_train_alt = []
    weights = []

    for i in range(0, len(X_train_raw), 2):
        X_train_alt.append(X_train_raw[i])
        Y_train_alt.append(Y_train_raw[i])

    for i in range(0, len(X_train_alt)):
        weights.append(i)

    plt.scatter(dates, prices, color="black", label="Data")
    svr_rbf = SVR(kernel="rbf", C=1e3, cache_size=7000, gamma = 0.1)
    svr_rbf.fit(X_train_alt, Y_train_alt, sample_weight=weights)

    svr_lin = SVR(kernel="linear", C=1e3, cache_size=7000)
    svr_lin.fit(X_train_alt, Y_train_alt, sample_weight=weights)

    svr_lin_nw = SVR(kernel="linear", C=1e3, cache_size=7000)
    svr_lin_nw.fit(X_train_alt, Y_train_alt)

    svr_rbf_nw = SVR(kernel="rbf", C=1e3, cache_size=7000, gamma = 0.1)
    svr_rbf_nw.fit(X_train_alt, Y_train_alt)

    plt.plot(X_test, svr_rbf.predict(X_test), color="green", label="RBF")
    plt.plot(X_test, svr_lin.predict(X_test), color="red", label="Linear")
    plt.plot(X_test, svr_rbf.predict(X_test), color="blue", label="RBF NW")
    plt.plot(X_test, svr_lin.predict(X_test), color="yellow", label="Linear NW")
    plt.legend()
    plt.show()

get_data("TSLA.csv")

# print(dates, prices)
# plot_data(dates, prices, "Open")
# print(len(dates), len(prices))
# train_data(dates, prices)
# train_data_alt(dates, prices)

def train_model2(dates, prices):
    dates = [[date] for date in dates]
    lin = linear_model.LinearRegression()
    x_train, x_test, y_train, y_test = train_test_split(dates, prices, test_size = 0.2, random_state = 4)
    print(len(x_train), len(y_train))
    lin.fit(x_train, y_train)
    print(lin.coef_)

# train_model2(dates, prices)
def line_of_best_fit(degree):
    plt.plot(dates, prices, 'o')
    p1 = sp.polyfit(dates, prices, degree)
    plt.plot(dates, sp.polyval(p1, dates), 'r-')
    prices_predict = p1[0] * dates + p1[1] # y = mx + b
    # print("Predicted:\n" , prices_predict, "Actual:\n", prices)
    plt.show()

line_of_best_fit(10)
