#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from scipy.interpolate import interp1d
from datetime import datetime
from pandas.plotting import register_matplotlib_converters
import seaborn as sns
from scipy.stats import spearmanr,skew,kurtosis,pearsonr
import matplotlib
import warnings
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats.mstats import winsorize
from scipy.stats import ttest_ind
from itertools import combinations
from datetime import datetime
import time
import itertools
from scipy import stats
from matplotlib.lines import Line2D
from matplotlib import rcParams, cycler
import matplotlib as mpl
import matplotlib.dates as mdate
import random

warnings.filterwarnings("ignore")

def date_strp_col(df):
    df.columns = [pd.to_datetime(str(i)) for i in df.columns.values]
    df = df.T.sort_index().T
    return df

def check_dup_rows(x):
    print(len(x))
    x1 = x.copy(deep=True)
    print(len(x.drop_duplicates()))
    return None

def check_col(df,s):
    return len(set(df[s].values))

def list_to_color(listx):
    re = []
    for i in listx:
        if i>=0:
            re.append("darkred")
        else:
            re.append("darkgreen")
    return re

def time_value():
    return time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())

def scale_df(df1):
    return pd.DataFrame(scale(df1),index = df1.index,columns = df1.columns)

def inx_col_intersec(df1_neu,df_close_loading):
    df_close_loading = date_strp_col(df_close_loading)
    df1_neu = date_strp_col(df1_neu)
    col_ = list(set(df1_neu.columns)&set(df_close_loading.columns))
    col_.sort()
    df1_neu = df1_neu[col_]
    df_close_loading = df_close_loading[col_]
    bingji = set(df1_neu.index.values)&set(df_close_loading.index.values)
    df1_neu = df1_neu.loc[bingji]
    df_close_loading = df_close_loading.loc[bingji]
    return df1_neu,df_close_loading