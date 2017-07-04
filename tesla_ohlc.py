import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_boston
import scipy as sp
import requests
import pandas_datareader.data as web
import bs4 as bs
import datetime as dt
import os
from os import walk

import pickle

al = 'all_aclose_data.csv'

style.use('ggplot')

def get_ohlc_data(filename):
    # reads data with date as index
    ohlc = pd.read_csv(filename, index_col=0, parse_dates=True)
    return ohlc

def graph_mavg(filename, price_type="Adj Close"):
    ohlc_data = get_ohlc_data(filename).resample('10D').mean()
    # print(ohlc_data)
    ohlc_data['100ma'] = ohlc_data[price_type].rolling(window=100, min_periods=0).mean()
    ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
    ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=1, colspan=1)

    ax1.plot(ohlc_data.index, ohlc_data[price_type])
    ax1.plot(ohlc_data.index, ohlc_data['100ma'])
    ax2.bar(ohlc_data.index, ohlc_data['Volume'])

    plt.show()

# graph_mavg('data/TSLA.csv')

# stores parsed data into one csv (all_aclose_data.csv)
def parse_all_data():
    files = []
    for (dirpath, dirnames, filenames) in walk('data/'):
        files.extend(filenames)
        break
    # get rid of .DS_Store
    files.pop(0)
    all_df = pd.DataFrame()

    for f in files:
        df = pd.read_csv("data/"+f)
        df.set_index('Date', inplace=True)
        df.rename(columns= {'Adj Close':f}, inplace=True)
        df.drop(['Open', 'Close', 'High', 'Low', 'Volume'], 1, inplace=True)

        if all_df.empty:
            all_df = df
        else:
            # accounts for different row values in each dataframe
            all_df = all_df.join(df, how='outer')

    if os.path.exists(al):
        os.remove(al)
    print(all_df.head())
    all_df.to_csv(al)

def read_s(filename):
    return pd.read_csv(filename)

# parse_all_data()
def graph_corr():
    df = read_s(al)
    df_corr = df.corr()
    data_array = df_corr.values
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    heatmap = ax.pcolor(data_array, cmap='RdYlGn')
    fig.colorbar(heatmap)
    ax.set_xticks(np.arange(data_array.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data_array.shape[1]) + 0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    column_labels = df_corr.columns
    row_labels = df_corr.index

    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap.set_clim(-1, 1)
    plt.show()

graph_corr()
