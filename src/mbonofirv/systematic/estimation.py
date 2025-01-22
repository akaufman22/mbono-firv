"""
This module provides functions for estimating butterfly and spread weights based on PCA of zero curves,
fitting Ornstein-Uhlenbeck parameters, loading historical data, generating trading signals, and plotting
strategy performance statistics.
"""
import sqlite3
from math import log, sqrt

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import statsmodels.api as sm
from src.mbonofirv.CurveCalc import define_ql_bonds_bulk
from src.mbonofirv.systematic.signals import MeanReversionSignal
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm


def get_pca_bf_weights(zc_data):
    """
    Estimates butterfly weights based on PCA of zero curve data.
    """
    cov = zc_data.cov()
    eig_vals, eig_vecs = np.linalg.eig(cov)
    permutation = np.argsort(-eig_vals)
    eig_vals = eig_vals[permutation]
    eig_vecs = eig_vecs[:, permutation]
    A = eig_vecs.T
    A[2] = [0, 1, 0]
    bf_weights = np.linalg.inv(A) @ np.array([[0], [0], [1]])
    return bf_weights

def get_pca_spread_weights(zc_data):
    """
    Estimates spread weights based on PCA of zero curve data.
    """
    cov = zc_data.cov()
    eig_vals, eig_vecs = np.linalg.eig(cov)
    permutation = np.argsort(-eig_vals)
    eig_vals = eig_vals[permutation]
    eig_vecs = eig_vecs[:, permutation]
    A = eig_vecs.T
    A[1] = [1, 0]
    spread_weights = np.linalg.inv(A) @ np.array([[0], [1]])
    return spread_weights

def fit_ou_params(x):
    """
    Fits Ornstein-Uhlenbeck parameters to a time series.
    """
    reg = sm.OLS(x[1:],sm.add_constant(x[:-1])).fit()
    theta = -log(reg.params[1])
    mu = reg.params[0] / (1 - reg.params[1])
    sigma_eq = np.std(reg.resid) / sqrt((1 - reg.params[1] ** 2))
    hl = log(2) / theta
    return theta, mu, sigma_eq, hl

def load_historical_data (db_path="./db/MBONOdata.db", generate_ql_objects=True, print_head=False):
    """
    Loads historical data from the database.
    """
    df_zc = pd.read_parquet('./db/zerocurve.parquet')
    con = sqlite3.connect(db_path)
    query = """SELECT DISTINCT BBGId, InstrumentID, Maturity, PricingDate from Instruments WHERE Maturity > 2010-01-01;"""
    df = pd.read_sql_query(query, con, parse_dates=['PricingDate', 'Maturity'])
    if generate_ql_objects is True:
        bonds = define_ql_bonds_bulk(list(df['BBGID'].values), id_type='BBGID', db_path=db_path)
        df['QL bond'] = bonds
    df.set_index('BBGID', inplace=True)
    query = """SELECT InstrumentID,  Date, Price FROM MarketData ;"""
    agg_price_data = pd.read_sql_query(query, con, parse_dates=['Date']).sort_values(['Date'])
    price_data = agg_price_data.pivot_table(index='Date', columns='InstrumentID', values='Price')
    price_data.columns = [df.index[df['InstrumentID'] == c][0] for c in price_data.columns]
    query = """SELECT InstrumentID,  Date, Residual FROM FitData ;"""
    agg_fit_data = pd.read_sql_query(query, con, parse_dates=['Date']).sort_values(['Date'])
    fit_data = agg_fit_data.pivot_table(index='Date', columns='InstrumentID', values='Residual')
    fit_data.columns = [df.index[df['InstrumentID'] == c][0] for c in fit_data.columns]
    if print_head:
        print('Curves')
        display(df_zc.head(5).style)
        print('Instruments')
        display(df.head(5).style)
        print('Prices')
        display(price_data.head(5).style)
        print('Fitted Residuals')
        display(fit_data.head(5).style)
    return df_zc, df, price_data, fit_data

def generate_triplet_signals(df_zc, tenors, lookback_years, test_years,
                     p_val_crit, hl_crit, sigma_crit,
                     z_ol, z_cl, z_os, z_cs):
    """
    Generates mean reversion signals based on PCA of zero curves.
    """
    eoy = pd.date_range(end=df_zc.index[-1], start = df_zc.index[0], freq='BY')
    pca_strategies = []
    for i in tqdm(range(lookback_years-1, len(eoy), 1)):
        idx = pd.bdate_range(start=eoy[i] + pd.tseries.offsets.DateOffset(years=-lookback_years), end=eoy[i])
        train_idx = df_zc.index.intersection(idx)
        idx = pd.bdate_range(end=eoy[i] + pd.tseries.offsets.DateOffset(years=test_years),
                            start=eoy[i] + pd.tseries.offsets.DateOffset(days=1))
        test_idx = df_zc.index.intersection(idx)
        for j, t1 in enumerate(tenors):
            for k, t2 in enumerate(tenors[j+1:j+2]):
                for t3 in tenors[j+k+2:j+k+3]:
                    pca_bf_weights = get_pca_bf_weights(df_zc.loc[train_idx, [t1, t2, t3]])
                    pca_bf_ts = df_zc.loc[train_idx, [t1, t2, t3]].dot(pca_bf_weights)
                    if adfuller(pca_bf_ts)[1] < p_val_crit:
                        _, mu, sigma_eq, hl  = fit_ou_params(pca_bf_ts.values)
                        if (hl < hl_crit) & (sigma_eq > sigma_crit):
                            print(test_idx[0], t1, t2, t3, int(hl), int(10000*sigma_eq))
                            pca_bf_test = df_zc.loc[test_idx, [t1, t2, t3]].dot(pca_bf_weights)
                            pca_strategy = MeanReversionSignal([t1, t2, t3], pca_bf_weights, pca_bf_test, mu, sigma_eq)
                            pca_strategy.set_trading_rule(ol=z_ol, cl=z_cl, os=z_os, cs=z_cs, ts=30)
                            pca_strategies.append(pca_strategy)
    return pca_strategies

def generate_spread_signals(df_zc, tenors, lookback_years, test_years,
                     p_val_crit, hl_crit, sigma_crit,
                     z_ol, z_cl, z_os, z_cs):
    """
    Generates mean reversion signals based on PCA of zero curves.
    """
    eoy = pd.date_range(end=df_zc.index[-1], start = df_zc.index[0], freq='BY')
    pca_spread_strategies = []
    for i in tqdm(range(lookback_years-1, len(eoy), 1)):
        idx = pd.bdate_range(start=eoy[i] + pd.tseries.offsets.DateOffset(years=-lookback_years), end=eoy[i])
        train_idx = df_zc.index.intersection(idx)
        idx = pd.bdate_range(end=eoy[i] + pd.tseries.offsets.DateOffset(years=test_years),
                            start=eoy[i] + pd.tseries.offsets.DateOffset(days=1))
        test_idx = df_zc.index.intersection(idx)
        for j, t1 in enumerate(tenors):
            for k, t2 in enumerate(tenors[j+1:j+5]):
                pca_spread_weights = get_pca_spread_weights(df_zc.loc[train_idx, [t1, t2]])
                pca_spread_ts = df_zc.loc[train_idx, [t1, t2]].dot(pca_spread_weights)
                if adfuller(pca_spread_ts)[1] < p_val_crit:
                    _, mu, sigma_eq, hl  = fit_ou_params(pca_spread_ts.values)
                    if (hl < hl_crit) & (sigma_eq > sigma_crit):
                        print(test_idx[0], t1, t2, int(hl), int(10000*sigma_eq))
                        pca_spread_test = df_zc.loc[test_idx, [t1, t2]].dot(pca_spread_weights)
                        strategy = MeanReversionSignal([t1, t2], pca_spread_weights, pca_spread_test, mu, sigma_eq)
                        strategy.set_trading_rule(ol=z_ol, cl=z_cl, os=z_os, cs=z_cs, ts=30)
                        pca_spread_strategies.append(strategy)
    return pca_spread_strategies

def plot_stats(total_pnl, trading):
    """
    Plots strategy performance statistics.
    """
    strat_stats = ['Avg GMV, $', 'Avg Ann PNL, $', 'Ann Std Dev, $', 'Return on GMV, %',
                'Daily Turnover Ratio, %',
                'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown, $', '1D VaR (95%), $']
    df_stats = pd.DataFrame(index=strat_stats, columns=['Before Costs', 'After Costs'])
    for i in range(2):
        df_stats.iloc[0, i] = (trading.position * trading.dirty_prices/100).abs().sum(axis=1).mean()
        df_stats.iloc[1, i] = total_pnl[i].mean() * 261
        df_stats.iloc[2, i] = total_pnl[i].std() * np.sqrt(261)
        df_stats.iloc[3, i] = df_stats.iloc[1, i] / df_stats.iloc[0, i] 
        df_stats.iloc[4, i] = (trading.trades * trading.dirty_prices).abs().sum(axis=1).sum() / \
            (trading.position * trading.dirty_prices).abs().sum(axis=1).sum()
        df_stats.iloc[5, i] = df_stats.iloc[1, i] / df_stats.iloc[2, i]
        df_stats.iloc[6, i] = df_stats.iloc[1, i] / np.sqrt(np.sum(total_pnl[i][total_pnl[i] < 0] ** 2) / len(total_pnl[i])) \
        / np.sqrt(261)
        df_stats.iloc[7, i] = (total_pnl[i].cumsum() - total_pnl[i].cumsum().cummax()).min()
        df_stats.iloc[8, i] = total_pnl[i].quantile(0.05)
    net_pnl = total_pnl[1]
    rolling_sharpe = np.sqrt(261) * (net_pnl.rolling(261).mean() / net_pnl.rolling(261).std())
    fig, ax = plt.subplots(4,1, figsize=(20,20), gridspec_kw={'height_ratios': [2, 1, 1, 1]}, sharex=True)
    ax[0].plot(net_pnl.cumsum(), label='Net PNL')
    ax[0].set_title('Strategy performance')
    ax[0].legend()
    ax[0].xaxis.set_label_text('Date')
    ax[0].xaxis.set_tick_params(labelbottom=True)
    ax[0].yaxis.set_label_text('PnL, $ (no compounding, no reinvestment)')
    ax[0].yaxis.set_major_formatter(ticker.EngFormatter(unit=''))
    ax[1].set_title('1 Year Rolling Sharpe Ratio')
    ax[1].plot(rolling_sharpe, label='Rolling SR')
    ax[1].xaxis.set_label_text('Date')
    ax[1].xaxis.set_tick_params(labelbottom=True)
    ax[1].yaxis.set_label_text('Sharpe Ratio')
    ax[1].axhline(df_stats.iloc[5,1], color='r', linestyle='--', label='Mean SR')
    ax[1].legend()
    ax[2].set_title('Daily Risk')
    ax[2].plot(net_pnl.rolling(261).quantile(0.05))
    ax[2].xaxis.set_label_text('Date')
    ax[2].xaxis.set_tick_params(labelbottom=True)
    ax[2].yaxis.set_label_text('95% 1D VaR')
    ax[2].yaxis.set_major_formatter(ticker.EngFormatter(unit=''))
    ax[3].set_title('Underwater Chart')
    ax[3].plot(net_pnl.cumsum() - net_pnl.cumsum().cummax())
    ax[3].xaxis.set_label_text('Date')
    ax[3].yaxis.set_label_text('Drawdown, $')
    ax[3].yaxis.set_major_formatter(ticker.EngFormatter(unit=''))

    return df_stats
