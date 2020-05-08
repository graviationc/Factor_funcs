from base_funcs import *

from trench_funcs import *


def annual_sharp(x,rf):
    return (np.mean(x) - rf/252)/np.std(x)*np.sqrt(252)

class factor():
    def __init__(self,factor_name,df_factor,df_close,group_number,intv):

        '''
        col_drop = df_factor.count()[df_factor.count()<group_number].index
        if len(col_drop)>0:
            print("Warning: ",len(col_drop)," columns don't have enough data")
            df_factor = df_factor.drop(col_drop,axis=1)
        '''
        df_factor = add_col(df_close,df_factor)

        self.factor_value,self.close_loading = inx_col_intersec(df_factor,df_close)
        self.group_number = group_number
        self.trade_interval = intv
        self.name = factor_name +"_" +time_value()

        self.df_ic = pd.DataFrame()
        self.ic_result = []
        self.quantile_rtn = pd.DataFrame()
        self.return_groups = []
        self.ls_re = []

    def ffill(self):
        self.factor_value = self.factor_value.fillna(method='ffill',axis=1)

    def print_ic(self):
        if len(self.ic_result)>0:
            return pd.DataFrame(self.ic_result,index=['Rank IC','Rank ICIR','PCT1','IC','ICIR','PCT2']).T
        else:
            return None

    def print_qtile_rtn(self):
        idx_ = ['LS RTN',"LS MaxDD"] + ['TOP'] + ['G{0}'.format(num + 2) for num in range(self.group_number - 2)] + ['BOTTOM']

        if len(self.return_groups)>0:
            return pd.DataFrame(self.return_groups,index=idx_).T
        else:
            return None



    def mining_corr(self,weather_plot):
        spr_corr_all_stocks = []
        df_rtn = self.close_loading.pct_change(self.trade_interval,axis=1).shift(-self.trade_interval,axis=1).copy(deep=True)
        df_ic = pd.DataFrame()
        df_ic["rank ic"] = self.factor_value.corrwith(df_rtn,method="spearman")
        df_factor_clip = self.factor_value.clip(lower=self.factor_value.quantile(0.05), upper=self.factor_value.quantile(0.95), axis=1).copy(deep=True)
        df_ic["ic"] = df_factor_clip.corrwith(df_rtn,method="pearson")

        df_ic.dropna(inplace=True)
        df_ic = df_ic.sort_index()
        intv1 = (df_ic.index[1] - df_ic.index[0]).days
        df_ic["rolling rank ic"] = df_ic["rank ic"].rolling(int(220/intv1)).mean()
        df_ic["rolling ic"] = df_ic['ic'].rolling(int(220/intv1)).mean()

        def corr_analysis(ic_list):
            x1 = np.nanmean(ic_list)  # IC
            x2 = x1/np.nanstd(ic_list) # ICIR
            x3 = len([i for i in ic_list if i<0])/len(ic_list)
            x3 = max(x3,1-x3)  #IC方向稳定性 
            return [x1,x2,x3]

        self.df_ic = df_ic.copy(deep=True)
        self.ic_result = corr_analysis(df_ic[["rank ic"]].values.flatten()) + corr_analysis(df_ic[["ic"]].values.flatten())

        if weather_plot == True:
            plt.figure(figsize=(40,4))
            ax = plt.subplot(141)

            xdate, ydata = df_factor_clip.count().index,df_factor_clip.count().values
            xlims = mdate.date2num([xdate[0], xdate[-1]])
            __, yv = np.meshgrid(np.linspace(0,1,200), np.linspace(0,1,200))
            ax.plot(xdate, ydata,'steelblue', linewidth=2)
            extent = [xlims[0], xlims[1], min(ydata)-10, max(ydata)+10]
            ax.imshow(yv, cmap=mpl.cm.Blues, origin='lower',alpha = 0.5, aspect = 'auto',extent = extent)
            ax.fill_between(xdate, ydata, max(ydata)+10, color='white')
            ax.spines['top'].set_visible(False) 
            ax.spines['right'].set_visible(False)

            ax11 = plt.subplot(142)
            ax11.set_prop_cycle(cycler('color', ['r', 'g', 'b']))
            dftemp = df_factor_clip.describe().T.iloc[:,[1,3,7]]
            ax11.plot(dftemp)
            ax11.legend(dftemp.columns, loc=2)

            plt.subplot(143)
            hist_ = [i for i in list(df_factor_clip.values.flatten()) if str(i)!="nan"]
            plt.hist(hist_,bins=30)

            plt.subplot(144)
            plt.bar(df_ic.index,df_ic['rank ic'],color=list_to_color(df_ic['rank ic'].values),width=6,alpha=0.5,label='rank ic') #,width=12
            plt.plot(df_ic.index,df_ic['rolling rank ic'],linewidth=4,label='rolling rank ic',c="b")

            ax=plt.gca()
            #plt.grid()
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')

            plt.legend()
            plt.show()
        
        return None


    def half_life(self,signal_plot):
        df_fac = self.factor_value.copy(deep=True)
        df = pd.DataFrame()
        for med in["spearman","pearson"]:
            df_icall = pd.DataFrame()
            if med=="pearson":
                df_fac = df_fac.clip(lower=df_fac.quantile(0.05), upper=df_fac.quantile(0.95), axis=1)
            for j in range(0,251,5):
                x = self.trade_interval + j
                df_rtn = self.close_loading.pct_change(self.trade_interval ,axis=1).shift(-x,axis=1).copy(deep=True)
                df_icall[j] = df_fac.corrwith(df_rtn,method=med)
            if med == "spearman":
                df["Rank IC"] = df_icall.mean()
            if med == "pearson":
                df["IC"] = df_icall.mean()
        # df.plot.area(stacked=False,alpha = 0.5)
        if signal_plot==True:
            plt.plot(df["Rank IC"],label="Rank IC",color='r')
            plt.plot(df["IC"],label="IC",color='g')
            plt.grid()
            plt.show()
        return df



    def quantile_profile(self,df_weight):
        x1 = self.factor_value.index
        x2 = self.factor_value.columns
        if type(df_weight)!=type(pd.DataFrame([1])):
            df_weight = pd.DataFrame(np.ones((len(x1),len(x2))),index=x1,columns=x2)
        
        order_days = self.close_loading.columns 
        group_name_list = ['TOP'] + ['G{0}'.format(num + 2) for num in range(self.group_number - 2)] + ['BOTTOM']
        df_all = pd.DataFrame()
        for j in range(self.trade_interval)[::2]:
            quantile_return_list = [[] for x in range(len(group_name_list))]


            col_all = []
            for trade_day_num in range(j,len(order_days)-self.trade_interval,self.trade_interval):
                trade_day = order_days[trade_day_num]
                next_trade_day_list = order_days[trade_day_num:trade_day_num+self.trade_interval+1]

                df_concat = pd.DataFrame()
                df_concat['factor_value'] = self.factor_value[trade_day].copy(deep=True)
                df_concat.dropna(inplace=True)

                if len(df_concat)>self.group_number:

                    col_all += order_days[trade_day_num+1:trade_day_num+1+self.trade_interval]

                    df_concat = df_concat.join(self.close_loading[next_trade_day_list].pct_change(axis=1).iloc[:,1:],how='left').copy(deep=True)
                    df_concat = df_concat.join(df_weight[next_trade_day_list[0]],how='left',rsuffix="weight").copy(deep=True)
                    df_concat = df_concat.replace(np.nan,0)

                    df_concat.sort_values(by='factor_value', ascending=True,inplace=True) # top 组合因子值小，bottom组合因子值大
                    idxs = trench_array(len(df_concat), self.group_number)

                    for iGroup in range(len(group_name_list)):
                        start_index = idxs[iGroup]
                        end_index = idxs[iGroup+1]
                        wei = [[i]*self.trade_interval for i in df_concat.iloc[start_index:end_index,-1].values]
                        v1 = df_concat.iloc[start_index:end_index,1:-1].copy(deep=True)

                        #quantile_return_list[iGroup] += list(np.average(v1,weights=wei,axis=0))
                        quantile_return_list[iGroup] += list(v1.mean().values)
                    
            df_qtile_rtn = pd.DataFrame(quantile_return_list, index=group_name_list,columns=col_all).T # order_days[j+1:trade_day_num+self.trade_interval+1]
            df_all = df_all.append(df_qtile_rtn).copy(deep=True)

        df_all = df_all.reset_index()

        df_all = df_all.groupby(df_all.columns[0]).mean()
        df_all = df_all.sort_index()
        self.quantile_rtn = df_all
        return self

    def neu_reg(self,df_nature_sec,df_mkt_size):

        df1 = self.factor_value.copy(deep=True)

        df_mkt_size = add_col(df1,df_mkt_size).fillna(method='ffill',axis=1)

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

        df_resid = dfre.pivot(index='index', columns='DATE', values='v').apply(pd.to_numeric)

        self.factor_value = add_col(self.factor_value,df_resid)

         

    def value_reg(self,df_mkt_size):

        df1 = self.factor_value.copy(deep=True)

        df_mkt_size = add_col(df1,df_mkt_size)
        df_mkt_size = df_mkt_size.fillna(method='ffill',axis=1)
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
        self.factor_value = dfre.pivot(index='index', columns='DATE', values='v').apply(pd.to_numeric)

    def dummy_reg(self,df_nature_sec):

        df1 = self.factor_value.copy(deep=True)

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
        self.factor_value = dfre.pivot(index='index', columns='DATE', values='v').apply(pd.to_numeric)


    def quntile_cumprod(self,w_plot):

        df_q = self.quantile_rtn.copy(deep=True)

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

        self.ls_re = ls_re.copy()

        ls_value = ls_re[0]

        index_i = np.argmax((np.maximum.accumulate(ls_value) - ls_value)/np.maximum.accumulate(ls_value))

        index_j = np.argmax(ls_value[:index_i])
        maxdd =  1 - ls_value[index_i]/ls_value[index_j]

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
        self.return_groups =  [pow(ls_value[-1],1/yyrs)-1,maxdd] + q_rtn
"""
df1 = pd.DataFrame(np.random.normal(size=(550,550)),columns = pd.date_range(start='1/1/2008', periods=550))
df2 = pd.DataFrame(np.random.normal(size=(550,550)),columns = pd.date_range(start='1/1/2008', periods=550))
f = factor('a',df1,df2,10,10)

re = f.quantile_profile(None)

quntile_cumprod(re,True)
"""

