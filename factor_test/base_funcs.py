#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from scipy.interpolate import interp1d
from datetime import datetime
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

def trench_array(list_num, num_gps):
    NumStks1 = int(list_num / num_gps)
    yushu1 = list_num % num_gps
    x = [NumStks1+1]*yushu1 + [NumStks1]*(num_gps - yushu1)
    random.shuffle(x)
    x = [0] + x
    return np.array(x).cumsum()

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
    return time.strftime("%Y_%m_%d", time.localtime())#_%H:%M:%S


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


def add_col(df_fac,df_value):
    df_fac = date_strp_col(df_fac)
    df_value = date_strp_col(df_value)

    add_dates = set(df_fac.columns) - set(df_value.columns)
    for ad in add_dates:
        df_value[ad] = np.nan
    df_value = df_value.T.sort_index().T
    return df_value.fillna(method = 'ffill',axis=1)

def factors_corr(df_list,w_plot):
    df_id = ["df" + str(i) for i in range(1,len(df_list)+1)]
    list2 = list(itertools.combinations(list(range(len(df_list))), 2))
    cor_all = []
    for id1,id2 in list2:

        df1 = df_list[id1]
        df2 = df_list[id2]
        df1 = date_strp_col(df1)
        df2 = date_strp_col(df2)
        df1,df2 = inx_col_intersec(df1,df2)
        df_re = pd.DataFrame()
        df_re["rank ic"] = df1.corrwith(df2,method="spearman")
        df_re["ic"] = df1.corrwith(df2,method="pearson")
        df_re = df_re.sort_index()
        n1 = round(df_re.mean().values[0],2)
        n2 = round(df_re.mean().values[1],2)
        titles = df_id[id1] + " "+ df_id[id2]+" rank ic:"+str(n1) +" ic:"+str(n2)

        if w_plot==1:
            plt.figure(figsize=(12,5))
            plt.plot(df_re)
            plt.legend(df_re.columns,bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
            plt.title(titles)
        cor_all.append([df_id[id1],df_id[id2],n1,n2])
        if w_plot==1:
            df_im = pd.DataFrame(cor_all,columns=["dfx","dfy","rank ic",'ic'])
    return df_im


def transition_matrix(df,group_num,intv):
    group_name_list = ['TOP'] + ['G{0}'.format(order_number + 2) for order_number in range(group_num - 2)] + ['BOTTOM']
    df_re = pd.DataFrame(index=group_name_list,columns=group_name_list)
    df_re = df_re.fillna(0)
    c = 0
    for num in range(0,len(df.columns)-intv,2):
        df_concat = df.iloc[:,[num,num+intv]].dropna().copy(deep=True)
        df_concat.columns = ["v1","v2"]

        group_stock_num = int(len(df_concat) / group_num)
        yushu = len(df_concat)%group_num
        group_stock_num_list = [group_stock_num+1]*yushu + [group_stock_num]*(group_num - yushu)
        group_list = []
        for i in range(group_num):
            group_list += group_stock_num_list[i]*[group_name_list[i]]
            
        df_concat = df_concat.sort_values("v1")
        df_concat["gp1"] = group_list
        df_concat = df_concat.sort_values("v2")
        df_concat["gp2"] = group_list
        df_count = df_concat.groupby(["gp1","gp2"]).count().reset_index()
        df_count = df_count.set_index("gp1")
        
        df_num_gp1 = df_concat.groupby("gp1").count()
        df_count = df_count.join(df_num_gp1,rsuffix="_right",how="left")
        df_count = df_count.reset_index()
        df_count["per"] = df_count["v1"]/df_count["v1_right"]
        
        re = df_count[["gp1","gp2","per"]].pivot(index='gp1', columns='gp2', values='per').copy(deep=True)
        re = re.fillna(0)
        df_re = df_re+re
        c+=1
    df_re = df_re/c
    df_re = df_re.loc[group_name_list][group_name_list]
    sns.heatmap(df_re)
    return df_re



