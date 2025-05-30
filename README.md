FIRV - Relative Value Framework for EM Local Debt and Systematic Trading Strategy
======

Relative Value framework for EM local debt which includes curve building, securities rich/cheap analysis, Principal Component Analysis of zero-coupon curve and statistical analysis of selected strategies. Here is a realization for Mex Peso denominated government bonds as an example. Calculated historical values are stored in sqlite database and live calculations are presented in form of plotly dashboard.
![Dashboard Demo](dashboarddemo.gif)

Systematic Trading Strategy
---------------------------

Brief summary for the strategy is [here](./Summary.pdf). The FIRV framework includes a systematic trading strategy module that allows users to backtest and implement trading strategies based on the relative value analysis. This module provides tools to:

- Define trading rules and signals based on the rich/cheap analysis.
- Backtest strategies using historical data stored in the sqlite database.
- Evaluate the performance of strategies using various metrics such as Sharpe ratio, drawdown, and cumulative returns.
- Visualize the results of the backtest using interactive plots.

[Notebook with the backtest](./backtest.ipynb)

[Comparison of trade construction methods](./SystematicButterflyTrading.ipynb)

![Strategy Performance](output.png)