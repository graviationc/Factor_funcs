# Factors

## [BAB](https://doi.org/10.1016/j.jfineco.2013.10.005) Betting against beta

Beta

## [CMH]([http://dx.doi.org/10.2139/ssrn.3018454](https://dx.doi.org/10.2139/ssrn.3018454)) Cold Minus Hot

trading volume

## [ETP](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2660598) Excess Tail Probability

$$ \int\limits_{1}^{\infty}f(x) dx-\int\limits_{-\infty}^{-1}f(x) dx   $$

use daily residuals $\varepsilon_{i,d}$ from month t to t - 11 as x:

$R_{i,d}=\alpha_{i}+\beta_{i}R_{m,d}+\gamma_{i} R_{m,d}^{2}+\varepsilon_{i,d}$

where $R_{i,d}$ is the excess return of stock i on day d, $R_{m,d}$ is the market excess return on day d, and $\varepsilon_{i,d}$ is the idiosyncratic return on day d.

## [1](http://www.cnki.com.cn/Article/CJFDTotal-JRYJ201907010.htm) [2](I: https://doi.org/10.1017/S0022109019000097) Realized Semivariance

在时间段$[0,T]$内，时间分成n等分，观察到$p_{0},p_{1},p_{n}$，n+1个对数价格。$I_{r}$为示性函数。

$RVOL=\sum_{i=1}^{n}r_{i}^{2},\qquad r_{i}=p_{i}-p_{i-1}$

$RV^{+}=\sum_{i=1}^{n}r_{i}^{2}\cdot I_{r_{i}>0}$

$RV^{-}=\sum_{i=1}^{n}r_{i}^{2}\cdot I_{r_{i}<0}$

$RSV=\frac{RV^{+}-RV^{-}}{RV}$

$r_{i}$序列的skewness： $RSK=\frac{\sqrt{n}\sum_{i=1}^{n}r_{i}^{3}}{RV^{3/2}}$

$r_{i}$序列的kurtosis：$RKT=\frac{n\sum_{i=1}^{n}r_{i}^{4}}{RV^{2}}$

- 不确定的好消息会导致股价上涨

- 为了获得好消息的不确定性，规避坏消息的不确定性，投资者要付出更高价格，接受更低期望收益。RSI与未来股票收益负相关
- 1文中采用T=1天，分成连续的5分钟，取过去

## [H](http://www.cnki.com.cn/Article/CJFDTotal-JRYJ201907011.htm) Short Term Herd Behavior

$B_{i,t}$为在t时段内个股i所有买单，$S_{i,t}$代表所有卖单，$p_{t}$为截面t所有股票买单占比均值。$AF_{i,t}$为假设投资者独立交易时，个股买单占比与所有股票平均买单占比偏离程度的期望值。假设$B_{i,t}\sim B(N_{i,t},p_{t})$。$HB_{i,t}$为买入羊群行为。$HS_{i,t}$为卖出羊群行为。

原文用每6秒加成的高频数据，成交价大于前一买一与前一卖一均价，为买单，小于则为卖单。

$N_{i,t}=B_{i,t}+S_{i,t}$

$BS_{i,t}=\frac{B_{i,t}}{B_{i,t}+S_{i,t}}-p_{t}$

$AF_{i,t}=\sum_{k=0}^{N_{i,t}}\begin{pmatrix}N_{i,t}\\ k\end{pmatrix} p_{t}^{k}(1-p_{t})^{N_{i,t}-k}\left | \frac{k}{N_{i,t}} -p_{t}\right |$

$H_{i,t}=|BS_{i,t}|-AF_{i,t}$

$HB_{i,t}=H_{i,t},\quad if\quad  BS_{i,t}>0 \quad else \quad 0$

$HS_{i,t}=H_{i,t},\quad if\quad  BS_{i,t}<0 \quad else \quad 0$

- 原文研究时间范围2009-2012
- 随着度量频率提高，羊群行为程度增加
- 短期羊群行为与信息不对称程度相关。日度换手率越低，机构投资者比率越高，大市值或小市值（相对中等市值），羊群行为更明显。
- 短期性羊群行为程度越大，投资者越可能忽略自己的信息，更多依靠他人行为决策，导致价格过度反应，发生反转。
- 由于投资者注意力有限，规模越大、交易越活跃的股票更易受到关注，这类股票羊群行为-买入之后更可能发生价格反转。
- 而对于规模越小、交易越不活跃的股票，信息环境质量更差，同等新的负面影响下，受到更大卖出压力，羊群行为-卖出之后更可能发生明显的价格反转。

## [RES](10.1093/rfs/hhz048) Resiliency

$R_{1}$是开盘后半小时（9点半至10点）涨跌幅，$R_{2}$是10点至收盘涨跌幅，取过去1个月数据计算RES，$\sigma$是日涨跌幅标准差。

$RES=\frac{Cov(R_{1},R_{2})}{\sigma}$

- investors demand a premium for less liquid stocks
- These authors (其他测量流动性方法) do not ﬁnd signiﬁcant evidence of an illiquidity premium when value weighted returns are used, or when micro-cap stocks are excluded.
- “resiliency” refers to the ability ﬁrst to withstand a shock, and then to recover from it
- if the market for a stock is not perfectly liquid, a liquidity shock will generate a transitory price movement that takes time to ==reverse==.
- The opening half-hour is a period when liquidity is of major concern.
- Regarding the Amihud measure, a large price change associated with small volume is taken to be a sign of an illiquid market.However, following the advent of public news, a large price change accompanied by small trading volume would evidence a more efﬁcient adjustment to a new equilibrium price
- Regarding the bid-ask spread, earlier orders can impose a negative externality on subsequent orders, thereby generating higher transaction costs that are not reﬂected in spreads.





# Research

##信用评级质量

- 与“发行人付费”模式相比，采用“投资人付费”资信给出的信用评级显著更低。 
- 这一结果并非由于“投资人付费”模式下评级机构 主观刻意地压低信用评级导致，而是与公司未来的盈利能力和违约风险相关，并得到 债券投资人认可。 这表明“投资人付费”模式下评级机构给出的信用评级质量更高。 
- “发行人付费”模式下，信用评级虽然可以在一定程度上包含公司的内部私有信息， 但由于独立性缺失问题，其总体的信用评级质量仍然低于“投资人付费”模式下的信用 评级质量。

##上市公司综合盈利水平与股票收益

- 行为金融错误定价理论认为，股票收益预测的来源是由于各种投资者的行为偏差所导致。 一般情况下，异质性风险 IVOL更高（股票的估值不确定性越大，信息不对称越严重），股票交易量越低（交易成本越高），流动性越差（潜在套利成本高），错误定价发生的概率增大。
- 基于投资摩擦的 Q 资产定价模型，当公司面临的投资摩擦水平越低时，公司盈利能力与股票预期收益的正向关系越强。对于资产规模较大的公司（能够向银行提供更多的抵押品），分红水平较高（未来现金流更好），国企（政府支持），投资摩擦水平更小。

##Persistence of Public Markets Manager Skill

- recent performance is not a good predictor of future results.

##You Can’t Always Trend When You Want

- three main factors that explain excess returns to a trendfollowing strategy: (1) the magnitude of market moves; (2) the strategy’s ability to translate a given market move into profits; (3) the degree of diversification within a trend-following portfolio.
- Our findings suggest that the lower returns in recent years are due to (1), not explained by (2) or (3)

##Time series momentum: Is it there?

- time series predictability is weak among the assets.

##Behavioural finance

- Overconfidence, Too much trading, Skill and luck
- Loss aversion, Fear of loss, sell winners and hold losers
- Inertia

##Stock and bond market interaction:Does momentum spill over?

- 股票市场散户多，债券市场都是机构（信息优势），散户更容易反应不足underreaction，所以债券市场应更好的反映基本面，更迅速发生变化

- 债券市场受关注度更低，可能导致债券市场不是信息有效informationally efﬁcient，交易不活跃，可能导致内幕知情人在股票市场获利，该逻辑下，股票市场更好反映基本面变化

- investment grade corporate bonds do not exhibit momentum at the three- to 12-month horizons.they exhibit reversals。

- signiﬁcant evidence exists of a momentum spillover from equities to investment grade corporate bonds of the same ﬁrm. Firms earning high (low) equity returns over the previous year earn high (low) bond returns the following year.

- The spillover results are stronger among ﬁrms with lower-grade debt and higher equity trading volume

- 溢出效应可能的传导机制：股价走高导致违约风险降低，评级提高，反之也成立。评级变化是滞后的，才有了溢出现象。

- the inclusion of contemporaneous ratings revisions reduces the predictive power of past stock returns. 

## Cross-asset signals and time series momentum

- 本文研究数据：不同国家的股债指数
- 股债市场同时出现盈利动量，未来一年经济未来基本面更好，工业产值更高，投资高，失业率下降。 反之也成立。
- a country’s past bond returns being positive predictors of the country’s future equity returns and of a country’s past equity returns being negative predictors of the country’s future bond returns.
- If (mutual fund) ﬂows affect contemporaneous returns and ﬂows chase performance, this provides one channel through which past bond and equity market performance can lead to return continuations and ==time series momentum==.
- 股市影响债市渠道：股市收益率影响央行制定货币政策工具federal funds rate。股市好，央行提高利率，债市承压。

## Factor Performance 2010-2019: A Lost Decade?

- fama三因子在2010-2019表现糟糕。相反，其他因子（low risk, price momentum, earnings momentum, analyst revisions, seasonals, and short-term reversal）表现较好。

## Momentum in Corporate Bond Returns

- 美国公司债存在动量效应，控制利率风险（与久期相关）、流动性（age、amount）、信用风险、评级变化后仍存在。
- 缺少交易量或较低的交易成本不是债券出现动量的原因
- private-ﬁrm 公司的非投资级债动量效应最强，可能原因是信息传递缓慢
- 根据过去一段时间收益率分组，评级更差的债更可能出现在winner或者loser
- 股债动量组合相关性低，存在股票对债券的动量溢出效应，但很大程度上是债券特有的动量效应

## Short- and Long-Horizon Behavioral Factors

- 短期行为误差例如；投资者注意力有限，会低估`underreact`季报中的盈利能力，在之后的几个季度才能修正错误定价。
- 长期行为误差例如：投资者对自己的私有信息过度自信，反应过激`overreact`，高估股价，导致未来反转。

## When Diversification Fails

- bonds diversify stocks during stock selloffs but stocks do not diversify bonds during bond selloffs.
- 美股涨的时候美股与International Equity 能diversify；美股跌的时候不能diversify，similar results across risk assets(hedge fund, high yield bond, MBS, EAFE stocks)
- Treasury Bond decouple from stocks in bad times and become positively correlated with stocks in good times.股债相关性与宏观经济相关。
- The strategy of managed volatility is a particularly effective and low-cost approach to overcome the failure of diversification.



