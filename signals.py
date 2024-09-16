import numpy as np
import pandas as pd
import QuantLib as ql

from tqdm import tqdm


class MeanReversionSignal():
    
    def __init__(self, tenors, weights, time_series, mu, sigma):
        self.tenors = tenors
        self.weights = weights.flatten()
        self.time_series = time_series
        self.mu = mu
        self.sigma = sigma
        self.target_risk = None
        
    def set_trading_rule(self, ol=-2, cl=-1, os=2, cs=1, ts=30):
        self.open_long = ol
        self.close_long = cl
        self.open_short = os
        self.close_short = cs
        self.time_stop = ts
        return self
    
    def estimate_target_risk(self):
        signal = (self.time_series - self.mu)/self.sigma
        discrete_signal = discretise_signal(signal, self.open_long, self.close_long, self.open_short, self.close_short)
        target_risk= np.repeat(self.weights.reshape(1,-1), len(signal), axis=0) * discrete_signal.to_numpy().reshape(-1,1)
        target_risk = pd.DataFrame(data=target_risk,
                                   index=self.time_series.index, columns=self.tenors)
        self.target_risk = target_risk
        return self
    
    def dumb_pnl(self, market_data):
        pnl = ((market_data.diff().shift(-1)) * self.target_risk).sum(axis=1)
        return pnl
    
class AggregatedSignal():
    
    def __init__(self, strategies, weights=None, dates=None):
        self.strategies = strategies
        if weights is None:
            weights = np.ones(len(strategies))
        self.weights = weights
        self.multiplier = 1
        self.target_risk = None
        self.dates = dates

    def estimate_target_risk(self):
        print('Aggregating Signals')
        if self.strategies[0].target_risk is None:
            self.strategies[0].estimate_target_risk()
        target_risk = self.strategies[0].target_risk * self.weights[0]
        for i in range(1, len(self.strategies)):
            if self.strategies[i].target_risk is None:
                self.strategies[i].estimate_target_risk()
            target_risk = target_risk.add(self.strategies[i].target_risk * self.weights[i],
                                            fill_value=0)
        target_risk.fillna(0, inplace=True)
        target_risk *= self.multiplier
        if self.dates is not None:
            target_risk = target_risk.reindex(self.dates, fill_value=0)
        self.target_risk = target_risk
        return self
    
    def dumb_pnl(self, market_data):
        pnl = ((market_data.diff().shift(-1)) * self.target_risk).sum(axis=1)
        return pnl
    
    def set_target_volatility(self, target_volatility, zc_curves):
        print('Setting Target Volatility')
        if self.target_risk is None:
            self.get_target_risk()
        self.target_volatility = target_volatility
        for d in tqdm(self.target_risk.index):
            d2 = d + pd.tseries.offsets.DateOffset(days=-1)
            d1 = d2 + pd.tseries.offsets.DateOffset(years=-1)
            idx = zc_curves.index.intersection(pd.bdate_range(d1, d2))
            rolling_history = zc_curves.diff().loc[idx, self.target_risk.columns].to_numpy()
            cov = np.cov(rolling_history.T)
            vol = np.sqrt(261) * np.sqrt(np.dot(np.dot(self.target_risk.loc[d].to_numpy(), cov), self.target_risk.loc[d].to_numpy().T))
            if vol != 0:
                self.target_risk.loc[d] = self.target_risk.loc[d] * target_volatility / vol
        return self
    
def discretise_signal(z_scores, ol=-2, cl=-1, os=2, cs=1):
    position = pd.Series(0, index=z_scores.index)
    position.iloc[0] = np.where(z_scores.iloc[0] <= ol, 1, np.where(z_scores.iloc[0] >= os, -1, 0))
    for i in range(1, len(z_scores.index)):
        position.iloc[i] = np.where(z_scores.iloc[i] <= ol, 1, np.where(z_scores.iloc[i] >= os, -1,
            np.where((z_scores.iloc[i] < cl) & (position.iloc[i-1] == 1), 1,
            np.where((z_scores.iloc[i] > cs) & (position.iloc[i-1] == -1), -1, 0))))
    return position


