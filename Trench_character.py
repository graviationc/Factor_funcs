#!/usr/bin/env python
# -*- coding: utf-8 -*- 

from Base_pkgs import *


def trench_array(list_num, num_gps):
    NumStks1 = int(list_num / num_gps)
    yushu1 = list_num % num_gps
    x = [NumStks1+1]*yushu1 + [NumStks1]*(num_gps - yushu1)
    random.shuffle(x)
    x = [0] + x
    return np.array(x).cumsum()


def quantile_character_value(df_fac,df_value,group_number,w_plot):
    col_drop = df_fac.count()[df_fac.count()<group_number].index
    if len(col_drop)>0:
        print(len(col_drop))
        df_fac = df_fac.drop(col_drop,axis=1)
    df_fac,df_value = inx_col_intersec(df_fac,df_value)

    re3 = []
    re5 = []
    for i in df_fac.columns:
        if i in df_value.columns:
            t = df_fac[[i]].join(df_value[[i]],how='left',rsuffix='_right').copy(deep=True).dropna()
            if len(t)>group_number:
                t.columns = ["fac","VALUE_1"]
                t = t.sort_values("fac")
                idxs = trench_array(len(t),group_number)
                re2 = []
                re4 = []
                for iGroup in range(group_number):
                    start_index = idxs[iGroup]
                    end_index = idxs[iGroup+1]
                    re1 = t["VALUE_1"].iloc[start_index:end_index].mean()
                    obsv = end_index - start_index
                    re2.append(re1)
                    re4.append(obsv)
                re3.append(re2)
                re5.append(re4)
    if w_plot==1:
        x = pd.DataFrame(re3).mean()
        plt.grid(linestyle="-.",axis='y',zorder=0)
        if x.min()>0:
            plt.ylim(x.min()*0.9,x.max()*1.01)
        else:
            if x.max()<0:
                plt.ylim(x.min()*1.01,x.max()*0.99)
            else:
                plt.ylim(x.min()*1.01,x.max()*1.01)

        plt.bar(list(np.array(list(x.index))+1),x,zorder=3)
        ax1=plt.gca()
        for j in ['left','right','top']:
            ax1.spines[j].set_visible(False)
        plt.show()
    return pd.DataFrame(re3).mean(),pd.DataFrame(re5).sum()
    
def quantile_portfolio(df_fac,df_close,df_value,gps1,gps2,intv,w_plot):
    col_drop = df_fac.count()[df_fac.count()<group_number].index
    if len(col_drop)>0:
        print(len(col_drop))
        df_fac = df_fac.drop(col_drop,axis=1)
    df_fac,df_value = inx_col_intersec(df_fac,df_value)

    df_rtn = df_close.pct_change(intv,axis=1).shift(-intv,axis=1).copy(deep=True)

    re3 = []
    for i in df_fac.columns[::intv][:-1]:
        t = df_fac[[i]].dropna().join(df_value[[i]],how='left',rsuffix='_VALUE').dropna().copy(deep=True)
        t = t.join(df_rtn[[i]],how='left',rsuffix='_RTN')
        t.columns = ["FACTOR","VALUE","RTN"]
        t = t.sort_values("VALUE")
        idxs = trench_array(len(t),gps1)
        re2 = []
        for iGroup in range(gps1):
            start_index = idxs[iGroup]
            end_index = idxs[iGroup+1]
            t_part = t.iloc[start_index:end_index,:].copy(deep=True)
            t_part = t_part.sort_values("FACTOR")

            re1 = []
            idxs2 = trench_array(len(t_part), gps2)
            for igp2 in range(gps2):
                sid2 = idxs2[igp2]
                eid2 = idxs2[igp2 + 1]
                re = t_part["RTN"].iloc[sid2:eid2].mean()
                re1.append(re)
            re2.append(re1)
        re3.append(re2)
    return re3
