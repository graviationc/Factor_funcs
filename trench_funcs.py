#!/usr/bin/env python
# -*- coding: utf-8 -*- 
from base_funcs import *

def quantile_character_value(df_fac,df_value,group_number,w_plot):
    df_fac = date_strp_col(df_fac)
    df_value = date_strp_col(df_value)
    df_value = add_col(df_fac,df_value)
    re3 = []
    re5 = []
    for i in df_fac.columns:
        t = df_fac[[i]].join(df_value[[i]],how='left',rsuffix='_right').copy(deep=True).dropna()
        if len(t)>=group_number:
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
    
def quantile_portfolio(df_fac,df_rtn,df_value,gps1,gps2,intv,w_plot,dot_or_add):
    df_fac = date_strp_col(df_fac)
    df_rtn = date_strp_col(df_rtn)
    df_value = date_strp_col(df_value)
 
    df_rtn = add_col(df_fac,df_rtn)
    df_value = add_col(df_fac,df_value)

    re3 = []
    for i in df_fac.columns[::intv]:
        t = df_fac[[i]].join(df_value[[i]],how='left',rsuffix='_VALUE').copy(deep=True)
        t = t.join(df_rtn[[i]],how='left',rsuffix='_RTN')
        t = t.dropna()
        if len(t)>=gps1*gps2:
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

    if dot_or_add=='dot':
        df_mul = pd.DataFrame(np.ones((gps1,gps2)))
        for j in re3:
            df_mul = (pd.DataFrame(j)+1).multiply(df_mul)
    else:
        df_mul = pd.DataFrame(np.zeros((gps1,gps2)))
        c = 0
        for j in re3:
            c+=1
            df_mul += pd.DataFrame(j)
        df_mul = df_mul/c

    df_mul.columns = [j+1 for j in df_mul.columns]
    df_mul.index = [j+1 for j in df_mul.index]

    if w_plot==1:

        df_mul.plot.bar(figsize=(5*gps1,5))
        plt.grid(linestyle="-.",axis='y')
        ax1=plt.gca()
        for j in ['left','right','top']:
            ax1.spines[j].set_visible(False)


        df_mul.T.plot.bar(figsize=(5*gps1,5))
        plt.grid(linestyle="-.",axis='y')
        ax1=plt.gca()
        for j in ['left','right','top']:
            ax1.spines[j].set_visible(False)
        plt.show()

    return df_mul





