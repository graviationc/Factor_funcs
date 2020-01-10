#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import random

def quantile_character_value(df_fac,df_value,group_number,w_plot):
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

def trench_array(list_num,num_gps):
    
    NumStks1 = int(list_num / num_gps)
    yushu1 = list_num%num_gps
    x = [NumStks1+1]*yushu1 + [NumStks1]*(num_gps - yushu1)
    random.shuffle(x)
    x = [0] + x
    return np.array(x).cumsum() 


def quantile_portfolio(df_fac,df_close,df_value,gp_num_1,gp_num_2,w_plot):
    col_drop = df_fac.count()[df_fac.count()<group_number].index
    if len(col_drop)>0:
        print(len(col_drop))
        df_fac = df_fac.drop(col_drop,axis=1)

    df_fac = df_fac.dropna(how='all',axis=1)
    df_fac = date_strp_col(df_fac)
    df_close = date_strp_col(df_close)
    df_fac,df_close = inx_col_intersec(df_fac,df_close)

    re3 = []
    for i in df_fac.columns:
        if i in df_value.columns:
            t = df_fac[[i]].dropna().join(df_value[[i]],how='left',rsuffix='_right').dropna().copy(deep=True)
            t.columns = ["FACTOR","VALUE"]
            t = t.sort_values("VALUE")
            idxs = trench_array(list_num,num_gps)
            for iGroup in range(gp_num_1):
                start_index = idxs[iGroup]
                end_index = idxs[iGroup+1]
                re1 = t.iloc[start_index:end_index]


    return None
