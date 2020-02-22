# Academia Factors

## BAB

Beta

## [CMH]([http://dx.doi.org/10.2139/ssrn.3018454](https://dx.doi.org/10.2139/ssrn.3018454)) Cold Minus Hot

trading volume

## [ETP](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2660598) Excess Tail Probability

$$\int\limits_{1}^{\infty}f(x) dx-\int\limits_{-\infty}^{-1}f(x) dx$$

use daily residuals $\varepsilon_{i,d}$ from month t to t - 11 as x:

$R_{i,d}=\alpha_{i}+\beta_{i}R_{m,d}+\gamma_{i} R_{m,d}^{2}+\varepsilon_{i,d}$

where $R_{i,d}$ is the excess return of stock i on day d, $R_{m,d}$ is the market excess return on day d, and $\varepsilon_{i,d}$ is the idiosyncratic return on day d.

## [RSI](http://www.cnki.com.cn/Article/CJFDTotal-JRYJ201907010.htm) Relative signed variation

在时间段$[0,T]$内，时间分成n等分，观察到$p_{0},p_{1},p_{n}$，n+1个对数价格。$I_{r}$为示性函数。

$RV=\sum_{i=1}^{n}r_{i}^{2},\qquad r_{i}=p_{i}-p_{i-1}$

$RV^{+}=\sum_{i=1}^{n}r_{i}^{2}\cdot I_{r_{i}>0}$

$RV^{-}=\sum_{i=1}^{n}r_{i}^{2}\cdot I_{r_{i}<0}$

$RSV=\frac{RV^{+}-RV^{-}}{RV}$

- 不确定的好消息会导致股价上涨

- 为了获得好消息的不确定性，规避坏消息的不确定性，投资者要付出更高价格，接受更低期望收益。RSI与未来股票收益负相关
- 原文采用T=1天，分成连续的5分钟

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

# Academia Research

## 信用评级质量

- 与“发行人付费”模式相比，采用“投资人付费”资信给出的信用评级显著更低。 
- 这一结果并非由于“投资人付费”模式下评级机构 主观刻意地压低信用评级导致，而是与公司未来的盈利能力和违约风险相关，并得到 债券投资人认可。 这表明“投资人付费”模式下评级机构给出的信用评级质量更高。 
- “发行人付费”模式下，信用评级虽然可以在一定程度上包含公司的内部私有信息， 但由于独立性缺失问题，其总体的信用评级质量仍然低于“投资人付费”模式下的信用 评级质量。



