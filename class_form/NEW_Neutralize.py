

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

    df_mkt_size = add_col(df1,df_mkt_size)

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
