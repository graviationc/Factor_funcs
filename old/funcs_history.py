
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

