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
from df_preview import *

warnings.filterwarnings("ignore")

def date_strp_col(df):
    df.columns = [pd.to_datetime(str(i)) for i in df.columns.values]
    df = df.T.sort_index().T
    return df
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

def half_life(df_fac,df_stk_adjclose_pivot,weather_plot,intv):
    df_fac,df_stk_adjclose_pivot = inx_col_intersec(df_fac,df_stk_adjclose_pivot)
    df = pd.DataFrame()
    for med in["spearman","pearson"]:
        df_icall = pd.DataFrame()
        if med=="pearson":
            df_fac = df_fac.clip(lower=df_fac.quantile(0.05), upper=df_fac.quantile(0.95), axis=1)
        for j in range(0,250,5):
            x = intv + j
            df_rtn = df_stk_adjclose_pivot.pct_change(intv,axis=1).shift(-x,axis=1).copy(deep=True)
            df_icall[j] = df_fac.iloc[:,::2].corrwith(df_rtn.iloc[:,::2],method=med)
        if med=="spearman":
            df["Rank IC"] = df_icall.mean()
        if med=="pearson":
            df["IC"] = df_icall.mean()
    #df.plot.area(stacked=False,alpha = 0.5)
    plt.plot(df["Rank IC"],label="Rank IC",color='r')
    plt.plot(df["IC"],label="IC",color='g')
    return None

def bar_rolling_plot(df,sp_1,sp_2):
    plt.bar(df.index,df[sp_1],color=list_to_color(df[sp_1].values),width=6,alpha=0.5,label=sp_1) #,width=12
    plt.plot(df.index,df[sp_2],linewidth=4,label=sp_2,c="b")
    return None

def mining_corr(df_factor_loading,df_close_x,weather_plot,weather_plot_2,intv):
    spr_corr_all_stocks = []
    df_rtn = df_close_x.pct_change(intv,axis=1).shift(-intv,axis=1).copy(deep=True)
    df_ic = pd.DataFrame()
    df_ic["rank ic"] = df_factor_loading.corrwith(df_rtn,method="spearman")
    df_factor_clip = df_factor_loading.clip(lower=df_factor_loading.quantile(0.05), upper=df_factor_loading.quantile(0.95), axis=1).copy(deep=True)
    df_ic["ic"] = df_factor_clip.corrwith(df_rtn,method="pearson")

    df_ic.dropna(inplace=True)
    df_ic = df_ic.sort_index()
    intv1 = (df_ic.index[1] - df_ic.index[0]).days
    df_ic["rolling rank ic"] = df_ic["rank ic"].rolling(int(220/intv1)).mean()
    df_ic["rolling ic"] = df_ic['ic'].rolling(int(220/intv1)).mean()
    df_ic_2 = df_ic.iloc[::,:].copy(deep=True)

    if weather_plot == 1:
        plt.figure(figsize=(50,4))
        ax = plt.subplot(151)

        xdate, ydata = df_factor_clip.count().index,df_factor_clip.count().values
        xlims = mdate.date2num([xdate[0], xdate[-1]])
        __, yv = np.meshgrid(np.linspace(0,1,200), np.linspace(0,1,200))
        ax.plot(xdate, ydata,'steelblue', linewidth=2)
        extent = [xlims[0], xlims[1], min(ydata)-10, max(ydata)+10]
        ax.imshow(yv, cmap=mpl.cm.Blues, origin='lower',alpha = 0.5, aspect = 'auto',extent = extent)
        ax.fill_between(xdate, ydata, max(ydata)+10, color='white')
        ax.spines['top'].set_visible(False) 
        ax.spines['right'].set_visible(False)


        ax11 = plt.subplot(152)
        ax11.set_prop_cycle(cycler('color', ['r', 'g', 'b']))
        dftemp = df_factor_clip.describe().T.iloc[:,[1,3,7]]
        ax11.plot(dftemp)
        ax11.legend(dftemp.columns, loc=2)

        plt.subplot(153)
        hist_ = [i for i in list(df_factor_clip.values.flatten()) if str(i)!="nan"]
        plt.hist(hist_,bins=30)

        plt.subplot(154)
        bar_rolling_plot(df_ic_2,"rank ic",'rolling rank ic')
        ax=plt.gca()
        #plt.grid()
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')

        plt.legend()

        plt.subplot(155)
        if weather_plot_2==1:
            half_life(df_factor_loading,df_close_x,weather_plot,intv)
            #plt.grid()
            plt.legend()
            ax=plt.gca()
            ax.spines['top'].set_visible(False) 
            ax.spines['right'].set_visible(False)
        plt.show()
    return df_ic

def corr_analysis(ic_list):
    x1 = np.nanmean(ic_list)  # IC
    x2 = x1/np.nanstd(ic_list) # ICIR
    x3 = len([i for i in ic_list if i<0])/len(ic_list)
    x3 = max(x3,1-x3)  #IC方向稳定性 
    return [x1,x2,x3]

def quantile_profile_3(trade_interval, group_num,df_factor_loading,df_close_x,df_weight):
    x1 = df_factor_loading.index
    x2 = df_factor_loading.columns
    if type(df_weight)==int:
        df_weight = pd.DataFrame(np.ones((len(x1),len(x2))),index=x1,columns=x2)
    
    order_days = df_factor_loading.columns 
    group_name_list = ['TOP'] + ['G{0}'.format(order_number + 2) for order_number in range(group_num - 2)] + ['BOTTOM']
    df_all = pd.DataFrame()
    for j in range(trade_interval)[::2]:
        quantile_return_list = [[] for x in range(len(group_name_list))]
        for trade_day_num in range(j,len(order_days)-trade_interval,trade_interval):
            trade_day = order_days[trade_day_num]
            next_trade_day_list = order_days[trade_day_num:trade_day_num+trade_interval+1]
            df_concat = pd.DataFrame()
            df_concat['factor_value'] = df_factor_loading[trade_day]
            df_concat.dropna(inplace=True)
            df_concat = df_concat.join(df_close_x[next_trade_day_list].pct_change(axis=1).iloc[:,1:],how='left')
            df_concat = df_concat.join(df_weight[next_trade_day_list[0]],how='left',rsuffix="weight")
            df_concat = df_concat.replace(np.nan,0)
            df_concat.sort_values(by='factor_value', ascending=True,inplace=True) # top 组合因子值小，bottom组合因子值大
            group_stock_num = int(len(df_concat) / group_num)
            yushu = len(df_concat)%group_num
            group_stock_num_list = [group_stock_num+1]*yushu + [group_stock_num]*(group_num - yushu)
            idxs = np.array(group_stock_num_list).cumsum()     

            for iGroup in range(len(group_name_list)):
                if iGroup==0:
                    start_index = 0
                else:
                    start_index = idxs[iGroup-1]
                end_index = idxs[iGroup]
                wei = [[i]*trade_interval for i in df_concat.iloc[start_index:end_index,-1].values]
                v1 = df_concat.iloc[start_index:end_index,1:-1].copy(deep=True)
                quantile_return_list[iGroup] += list(np.average(v1,weights=wei,axis=0))
                
        df_qtile_rtn = pd.DataFrame(quantile_return_list, index=group_name_list,columns=order_days[j+1:trade_day_num+trade_interval+1]).T
        df_all = df_all.append(df_qtile_rtn)

    df_all = df_all.reset_index()
    df_all = df_all.groupby(df_all.columns[0]).mean()
    df_all = df_all.sort_index()
    return df_all

def factor_exposure_mkt(df1,df_mkt_size,group_num,w_plot):
    df1 = date_strp_col(df1)
    df_mkt_size = date_strp_col(df_mkt_size)
    df1,df_mkt_size = inx_col_intersec(df1,df_mkt_size)
    df_mkt_tile = df_mkt_size.rank(method='average', na_option='keep', ascending=True, pct=True).copy(deep=True)
    group_name_list = ['TOP'] + ['G{0}'.format(order_number + 2) for order_number in range(group_num - 2)] + ['BOTTOM']
    re_all = []
    for day in df1.columns:
        df_concat = df1[[day]].copy(deep=True)
        df_concat.columns = ["factor_value"]
        df_concat.dropna(inplace=True)
        df_concat.sort_values(by='factor_value', ascending=True,inplace=True) # top 组合因子值小，bottom组合因子值大
        group_stock_num = int(len(df_concat) / group_num)
        re_daily = []
        for iGroup in range(len(group_name_list)):
            start_index = iGroup * group_stock_num
            end_index = (iGroup + 1) * group_stock_num                    
            if iGroup == len(group_name_list) - 1:
                end_index = len(df_concat)
            group_stocks = df_concat.iloc[start_index:end_index].index
            re = df_mkt_tile[[day]].loc[group_stocks].mean().values[0]
            re_daily.append(re)
        re_all.append([day] + re_daily)
    df_re_all = pd.DataFrame(re_all)
    df_re_all = df_re_all.set_index(0)
    df_re_all.columns = group_name_list
    df_re_all = df_re_all.sort_index()

    if w_plot==1:
        plt.figure(figsize=(24,5))
        ax = plt.subplot(121)
        ax.plot(df_re_all[['TOP']],color='darkblue')
        ax.plot(df_re_all[['BOTTOM']],color='darkred')
        ax.legend(['TOP','BOTTOM'],loc=2)

        ax2 = plt.subplot(122)
        cmap = plt.cm.coolwarm
        ax2.set_prop_cycle(cycler(color=cmap(np.linspace(0, 1, len(df_re_all.columns)))))
        ax2.plot(df_re_all)
        ax2.legend(df_re_all.columns,bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0,ncol=2)
    return None

def quntile_cumprod(df_q,w_plot):
    start_date = df_q.index[0] - (df_q.index[1] - df_q.index[0])
    temp = pd.DataFrame([[0]*len(df_q.columns)],columns=df_q.columns,index=[start_date])
    df_q = temp.append(df_q)
    df_q = df_q.replace(np.nan,0)
    allq = (df_q+1).cumprod().copy(deep=True)  

    y_train = allq.iloc[-1,:].values
    x_train = np.array(list(range(1,len(df_q.columns)+1))).reshape(-1,1)
    lr = LinearRegression()
    lr.fit(x_train,y_train)

    if lr.coef_ > 0:
        sig = -1
    else:
        sig = 1

    num1 = int(len(df_q.columns)/10)
    num2 = int(len(df_q.columns)/10)*3
    num3 = int(len(df_q.columns)/10)*5
    new_num_list = list([i for i in set([1,num1,num2,num3]) if i>0])

    ls_re = []
    for i in new_num_list[:1]:
        df_ls = df_q.iloc[:,:i].join(df_q.iloc[:,-i:]).copy(deep=True)  
        df_ls.iloc[:,:i] = sig*df_ls.iloc[:,:i]
        df_ls.iloc[:,-i:] = -1*sig*df_ls.iloc[:,-i:]

        df_ls["ls"] = df_ls.mean(axis=1)
        df_ls_ = (df_ls+1).cumprod().copy(deep=True) 
        ls_re.append(df_ls_["ls"].values)
    ls_value = ls_re[0]

    index_j = np.argmax(np.maximum.accumulate(ls_value) - ls_value)
    index_i = np.argmax(ls_value[:index_j])      
    maxdd =  1 - ls_value[index_i]/ls_value[index_j]  # 最大回撤

    t1 = pd.to_datetime(allq.index.values[0])
    t2 = pd.to_datetime(allq.index.values[-1])
    yyrs = (t2-t1).days/365
    q_rtn = []
    for j in allq.values[-1]:
        q_rtn.append(pow(j,1/yyrs)-1)
    if w_plot==1:
        #matplotlib.rc('xtick', labelsize=13)
        #matplotlib.rc('ytick', labelsize=20) 
        #xax = [datetime.strptime(str(i), '%Y-%m-%d') for i in allq.index]
        xax =  allq.index.values
        plt.figure(figsize=(40,4))

        plt.subplot(141)
        plt.bar(range(len(allq.columns)), q_rtn, color=list_to_color(q_rtn), tick_label=allq.columns)
        ax=plt.gca()
        ax.spines['top'].set_visible(False) 
        ax.spines['right'].set_visible(False)
        ax22 = plt.subplot(142)
        ax22.spines['top'].set_visible(False)
        ax33 = ax22.twinx()
        ax33.spines['top'].set_visible(False)

        m_rtn = (df_ls["ls"]+1).cumprod()[::20].copy(deep=True)
        m_rtn = m_rtn.pct_change(1).replace(np.nan,0)

        ax33.bar(m_rtn.index,m_rtn,color=list_to_color(m_rtn.values),width=20,label="monthly return")
        ax22.set_prop_cycle(cycler('color', ['r', 'g', 'b','k']))
        if allq.iloc[:,0].values[-1]>allq.iloc[:,-1].values[-1]:
            c1,c2 = "darkred","g"
        else:
            c2,c1 = "darkred","g"
        for i in range(len(new_num_list))[:1]:
            if i==0:
                label1 = 'LongShort'
            else:
                label1 = 'LongShort '+str(new_num_list[i]/len(df_q.columns))
            ax22.plot(xax,ls_re[i],label=label1,c="b")
        ax22.legend(loc='upper left')
        ax33.legend(loc='upper right')

        plt.subplot(143)
        plt.plot(xax,allq.iloc[:,0].values,label = allq.columns[0],c=c1)
        plt.plot(xax,allq.iloc[:,-1].values,label = allq.columns[-1],c=c2)
        plt.legend(loc='upper right')
        ax=plt.gca()
        ax.spines['top'].set_visible(False) 
        ax.spines['right'].set_visible(False)
        
        cmap = plt.cm.coolwarm
        rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, len(allq.columns))))
        ax = plt.subplot(144)
        custom_lines = [Line2D([0], [0], color=cmap(0.), lw=4),
                        Line2D([0], [0], color=cmap(.5), lw=4),
                        Line2D([0], [0], color=cmap(1.), lw=4)]
        for i in range(len(allq.columns)):
            ax.plot(xax,allq.iloc[:,i].values)#,label = allq.columns[i])
        ax.legend(custom_lines,['TOP', 'G10', 'BOTTOM'], loc=2)
        ax1=plt.gca()
        ax1.spines['top'].set_visible(False) 
        ax1.spines['right'].set_visible(False)

        plt.show()
    return [pow(ls_value[-1],1/yyrs)-1,maxdd] + q_rtn
        
def scale_df(df1):
    return pd.DataFrame(scale(df1),index = df1.index,columns = df1.columns)


def neu_reg_fast(df1,df_nature_sec,df_mkt_size):
    df1,df_mkt_size = inx_col_intersec(df1,df_mkt_size)
    df_nature_sec = df_nature_sec.dropna()
    bingji = set(df1.index.values)&set(df_nature_sec.index.values)
    df1 = df1.loc[bingji]
    df_nature_sec = df_nature_sec.loc[bingji]
    df_mkt_size = df_mkt_size.loc[bingji]
    
    df_mkt_size = df_mkt_size.clip(lower=df_mkt_size.quantile(0.01), upper=df_mkt_size.quantile(0.99), axis=1)
    df_mkt_size = scale_df(df_mkt_size)

    df_nature_dummy = pd.DataFrame(index = df_nature_sec.index)
    for i in df_nature_sec.columns:
        tmp = pd.get_dummies(df_nature_sec[i])
        df_nature_dummy = df_nature_dummy.join(tmp,how='left')
    df1 = scale_df(df1)
    reall1,reall2,reall3=[],[],[]
    for i in range(len(df1.columns)):
        col_ = df1.columns[i]
        y1 = df1.iloc[:,i].values.flatten()
        x1 = df_mkt_size[col_].values.flatten()
        sel = ~np.isnan(x1) & ~np.isnan(y1)
        x1 = x1[sel]
        y1 = y1[sel]
        num =len(y1)
        if num>10: 
            x2 = df_nature_dummy.loc[df1.index[sel]].values
            x3 = np.concatenate((x2,x1.reshape(-1,1) ), axis=1)
            #a = np.concatenate((x3,np.ones(len(x3)).reshape(-1,1) ), axis=1)
            xx = np.linalg.lstsq(x3, y1.reshape(-1,1), rcond=None)            
            pred = [np.dot(i,xx[0])[0] for i in x3]
            re = np.array(y1) - np.array(pred)
            reall1+= list(re)
            reall2+= list(df1.index[sel])
            reall3+= [col_]*num
        else:
            pass
    allv = np.array([reall1,reall2,reall3]).T
    dfre = pd.DataFrame(allv,columns=["v","index","DATE"])
    return dfre.pivot(index='index', columns='DATE', values='v').apply(pd.to_numeric)


def value_reg_fast(df1,df_mkt_size):
    df1,df_mkt_size = inx_col_intersec(df1,df_mkt_size)
    df1 = scale_df(df1)
    df_mkt_size = df_mkt_size.clip(lower=df_mkt_size.quantile(0.01), upper=df_mkt_size.quantile(0.99), axis=1)
    df_mkt_size = scale_df(df_mkt_size)

    reall1,reall2,reall3=[],[],[]
    for i in range(len(df1.columns)):
        col_ = df1.columns[i]
        y1 = df1.iloc[:,i].values.flatten()
        x1 = df_mkt_size[col_].values.flatten()
        sel = ~np.isnan(x1) & ~np.isnan(y1)
        x1 = x1[sel]
        y1 = y1[sel]
        num =sum(sel)
        if num>10:   
            slope, intercept, r_value, p_value, std_err = stats.linregress(x1,y1)
            re = y1 - x1*slope - intercept
            reall1+= list(re)
            reall2+= list(df1.index[sel])
            reall3+= [col_]*num
        else:
            #reall1+= [np.nan]*len(df1.index)
            #reall2+= list(df1.index)
            #reall3+= [col_]*len(df1.index)
            pass
    allv = np.array([reall1,reall2,reall3]).T
    dfre = pd.DataFrame(allv,columns=["v","index","DATE"])
    return dfre.pivot(index='index', columns='DATE', values='v').apply(pd.to_numeric)

def dummy_reg_fast(df1,df_nature_sec):
    df1 = date_strp_col(df1)
    df_nature_sec = df_nature_sec.dropna()
    bingji = set(df1.index.values)&set(df_nature_sec.index.values)
    df1 = df1.loc[bingji]
    df_nature_sec = df_nature_sec.loc[bingji]

    df1 = scale_df(df1)
    df_nature_dummy = pd.DataFrame(index = df_nature_sec.index)
    for i in df_nature_sec.columns:
        tmp = pd.get_dummies(df_nature_sec[i])
        df_nature_dummy = df_nature_dummy.join(tmp,how='left')
    reall1,reall2,reall3=[],[],[]
    for i in range(len(df1.columns)):
        col_ = df1.columns[i]
        y1 = df1.iloc[:,i].dropna().copy(deep=True)
        num =len(y1)
        if num>10:  
            stks = y1.index
            x1 = df_nature_dummy.loc[stks].values
            a = np.concatenate((x1,np.ones(len(x1)).reshape(-1,1) ), axis=1)
            xx = np.linalg.lstsq(a, y1.values, rcond=None)
            slope1= xx[0][:-1]
            intercept1= xx[0][-1] 
            re = y1.values - np.array([np.dot(i,slope1)+intercept1 for i in x1])
            reall1+= list(re)
            reall2+= list(stks)
            reall3+= [col_]*num
        else:
            #reall1+= [np.nan]*len(df1.index)
            #reall2+= list(df1.index)
            #reall3+= [col_]*len(df1.index)
            pass

    allv = np.array([reall1,reall2,reall3]).T
    dfre = pd.DataFrame(allv,columns=["v","index","DATE"])
    return dfre.pivot(index='index', columns='DATE', values='v').apply(pd.to_numeric)

def factor_quantile_test(df1_neu,df_close_loading,df_weight,group_number,weather_plot,weather_plot_2,intv):

    df1_neu = df1_neu.drop(df1_neu.count()[df1_neu.count()<group_number].index,axis=1)

    df1_neu = df1_neu.dropna(how='all',axis=1)
    df1_neu = date_strp_col(df1_neu)
    df_close_loading = date_strp_col(df_close_loading)
    df1_neu,df_close_loading = inx_col_intersec(df1_neu,df_close_loading)

    df_ic = mining_corr(df1_neu,df_close_loading,weather_plot,weather_plot_2,intv)  
    result_corr = corr_analysis(df_ic[["rank ic"]].values.flatten()) + corr_analysis(df_ic[["ic"]].values.flatten())

    if weather_plot==0:
        return result_corr,df_ic
    else:
        x1 = quantile_profile_3(intv,group_number,df1_neu,df_close_loading,df_weight)
        pingjia_list = result_corr + quntile_cumprod(x1,weather_plot)
        plt.figure(figsize=(40,4))
        count=1
        for j in [20,60,120,240]:
            plt.subplot(1,4,count)
            transition_matrix(df1_neu,group_number,j)
            count+=1
        return pingjia_list,df_ic


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

def qtile_character(df_fac,df_value,group_number,w_plot):
    re3 = []
    for i in df_fac.columns:
        if i in df_value.columns:
            t = df_fac[[i]].dropna().join(df_value[[i]],how='left',rsuffix='_right').dropna()
            t.columns = ["fac","mkt"]
            t = t.sort_values("fac")
            group_stock_num = int(len(t) / group_number)
            yushu = len(t)%group_number
            group_stock_num_list = [group_stock_num+1]*yushu + [group_stock_num]*(group_number - yushu)
            idxs = np.array(group_stock_num_list).cumsum() 
            re2 = []
            for iGroup in range(group_number):
                if iGroup==0:
                    start_index = 0
                else:
                    start_index = idxs[iGroup-1]
                end_index = idxs[iGroup]
                re1 = t["mkt"].iloc[start_index:end_index].mean()
                re2.append(re1)
            re3.append(re2)
    if w_plot==1:
        x = pd.DataFrame(re3).mean()
        plt.grid(linestyle="-.",axis='y')
        plt.bar(x.index,x)
        plt.ylim(x.min()*0.99,x.max()*1.01)
        ax1=plt.gca()
        for j in ['left','right','top']:
            ax1.spines[j].set_visible(False) 
        plt.show()
    return pd.DataFrame(re3).mean()


"""
def value_regression(df1,df_mkt_size):
    df1,df_mkt_size = inx_col_intersec(df1,df_mkt_size)
    df1 = scale_df(df1)
    df = df_mkt_size
    df_mkt_size = df.clip(lower=df.quantile(0.01), upper=df.quantile(0.99), axis=1)
    df_mkt_size = scale_df(df_mkt_size)

    df_cancha = pd.DataFrame(index = df1.index) 
    for i in range(len(df1.columns)):
        df_temp = df1.iloc[:,i].dropna()
        df_temp = pd.DataFrame(df_temp).join(df_mkt_size[df1.columns[i]],rsuffix="xc")
        df_temp = df_temp.dropna()
        if len(df_temp)>10:
            lr_x = df_temp.iloc[:,1].values.reshape(-1, 1)
            lr_y = df_temp.iloc[:,0].values
            reg = LinearRegression().fit(lr_x,lr_y)
            df_temp['cancha'+str(i)] = lr_y - reg.predict(lr_x)
            df_cancha = df_cancha.join(df_temp['cancha'+str(i)])
        else:
            df_cancha = df_cancha.join(df1.iloc[:,i])
    df_cancha.columns = df1.columns
    return df_cancha  

def dummy_regression(df1,df_nature_sec):
    df_nature_dummy = pd.DataFrame(index = df_nature_sec.index)
    for i in df_nature_sec.columns:
        tmp = pd.get_dummies(df_nature_sec[i])
        df_nature_dummy = df_nature_dummy.join(tmp,how='left')

    df1 = scale_df(df1)
    df_cancha = pd.DataFrame(index = df1.index) 
    for i in range(len(df1.columns)):
        df_temp = df1.iloc[:,i].dropna()
        df_temp = pd.DataFrame(df_temp).join(df_nature_dummy)
        df_temp = df_temp.dropna()
        if len(df_temp)>10:
            lr_x = df_temp.iloc[:,1:].values
            lr_y = df_temp.iloc[:,0].values
            reg = LinearRegression()
            reg.fit(lr_x,lr_y)
            df_temp['cancha'+str(i)] = lr_y - reg.predict(lr_x)
            df_cancha = df_cancha.join(df_temp['cancha'+str(i)])
        else:
            df_cancha = df_cancha.join(df1.iloc[:,i])
    df_cancha.columns = df1.columns
    return df_cancha
def neu_regression(df1,df_nature_sec,df_mkt_size):
    df1,df_mkt_size = inx_col_intersec(df1,df_mkt_size)
    df1 = scale_df(df1)
    if type(df_mkt_size)!=int:
        df = df_mkt_size
        df_mkt_size = df.clip(lower=df.quantile(0.01), upper=df.quantile(0.99), axis=1)
        df_mkt_size = scale_df(df_mkt_size)
    
    if type(df_nature_sec)!=int:
        df_nature_dummy = pd.DataFrame(index = df_nature_sec.index)
        for i in df_nature_sec.columns:
            tmp = pd.get_dummies(df_nature_sec[i])
            df_nature_dummy = df_nature_dummy.join(tmp,how='left')

    df_cancha = pd.DataFrame(index = df1.index) 
    for i in range(len(df1.columns)):
        df_temp = df1.iloc[:,i].dropna()
        
        df_temp = pd.DataFrame(df_temp)
        if type(df_mkt_size)!=int:
            df_temp = df_temp.join(df_mkt_size[df1.columns[i]],rsuffix="xc",how='left')
        if type(df_nature_sec)!=int:
            df_temp = df_temp.join(df_nature_dummy,rsuffix="xc",how='left')

        df_temp = df_temp.dropna()
        if len(df_temp)>10:
            lr_x = df_temp.iloc[:,1:].values
            lr_y = df_temp.iloc[:,0].values
            reg = LinearRegression(fit_intercept=False).fit(lr_x,lr_y)
            df_temp['cancha'+str(i)] = lr_y - reg.predict(lr_x)
            df_cancha = df_cancha.join(df_temp['cancha'+str(i)])
        else:
            df_cancha = df_cancha.join(df1.iloc[:,i])
    df_cancha.columns = df1.columns
    return df_cancha
def quantile_profile_2(trade_interval, group_num,df_factor_loading,df_close_x):
    order_days = df_factor_loading.columns 
    group_name_list = ['TOP'] + ['G{0}'.format(order_number + 2) for order_number in range(group_num - 2)] + ['BOTTOM']

    df_all = pd.DataFrame()
    for j in range(trade_interval):
        quantile_return_list = [[] for x in range(len(group_name_list))]
        for trade_day_num in range(j,len(order_days)-trade_interval,trade_interval):
            trade_day = order_days[trade_day_num]
            next_trade_day_list = order_days[trade_day_num:trade_day_num+trade_interval+1]
            df_concat = pd.DataFrame()
            df_concat['factor_value'] = df_factor_loading[trade_day]
            df_concat.dropna(inplace=True)
            df_concat = df_concat.join(df_close_x[next_trade_day_list].pct_change(axis=1).iloc[:,1:],how='left')
            #df_concat = df_concat.join(df_weight[next_trade_day_list[0]],how='left',rsuffix="weight")
            df_concat = df_concat.replace(np.nan,0)
            df_concat.sort_values(by='factor_value', ascending=True,inplace=True) # top 组合因子值小，bottom组合因子值大
            group_stock_num = int(len(df_concat) / group_num)
            for iGroup in range(len(group_name_list)):
                start_index = iGroup * group_stock_num
                end_index = (iGroup + 1) * group_stock_num
                if iGroup == len(group_name_list) - 1:
                    end_index = len(df_concat)
                v = list(df_concat.iloc[start_index:end_index,1:].mean().values)
                quantile_return_list[iGroup] += v
        df_qtile_rtn = pd.DataFrame(quantile_return_list, index=group_name_list,columns=order_days[j+1:trade_day_num+trade_interval+1]).T
        df_all = df_all.append(df_qtile_rtn)

    df_all = df_all.reset_index()
    df_all = df_all.groupby(df_all.columns[0]).mean()
    df_all = df_all.sort_index()
    return df_all

"""

