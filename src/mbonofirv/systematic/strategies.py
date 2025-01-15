import pandas as pd
import numpy as np
import QuantLib as ql

from tqdm import tqdm

class SystematicStrategy():
    
    def __init__ (self, strategy, instrument_universe, market_data, zc_curves,
                  rebal_threshold=0, trading_threshold=0, time_slippage=0, fit_resid=None,
                  rebal_rule='vol', calendar=ql.Mexico(), day_count=ql.Actual360(),
                  settlement_days=2, yield_basis=ql.Semiannual):
        self.strategy = strategy
        self.instrument_universe = instrument_universe
        self.market_data = market_data
        self.zc_curves = zc_curves
        self.rebal_threshold = rebal_threshold
        self.trading_threshold = trading_threshold
        self.time_slippage = time_slippage
        self.tenors = self.zc_curves.columns
        self.tenors_years = [int(s[1:]) for s in self.tenors]
        self.fit_resid = fit_resid
        self.rebal_rule = rebal_rule
        self.calendar = calendar
        self.day_count = day_count
        self.settlement_days = settlement_days
        self.yield_basis = yield_basis
        
    def get_ql_obj(self, date):
        ql_date = ql.Date(date.day, date.month, date.year)
        ql.Settings.instance().evaluationDate = ql_date
        zero_curve = self.zc_curves.loc[date]
        spot_rates = [zero_curve.values[0]] + list(zero_curve.values)
        spot_dates = [ql_date + ql.Period(int(t), ql.Years) for t in ([0] + self.tenors_years)]
        spot_curve = ql.ZeroCurve(spot_dates, spot_rates, self.day_count, self.calendar)
        spot_curve_handle = ql.YieldTermStructureHandle(spot_curve)
        return spot_curve_handle

    def estimate_risk(self, position, date, spot_curve_handle):
        risk = pd.Series(index=self.tenors,
                         data=portfolio_risk(self.instrument_universe, date, position, spot_curve_handle,
                              self.tenors_years))
        return risk
    
    def order_to_open(self, date, risk, ql_curve_handle):
        if self.fit_resid is None:
            fit_data = None
        else:
            fit_data = self.fit_resid.loc[date]
        order = trading_order(self.instrument_universe, date, risk, ql_curve_handle,
                              self.tenors_years, fit_data=fit_data)
        return order    
    
    def rebalance_order(self, position, target_risk, date, spot_curve_handle):
        position_risk = self.estimate_risk(position, date, spot_curve_handle)
        tracking_error = position_risk - target_risk
        d2 = date + pd.tseries.offsets.DateOffset(days=-1)
        d1 = d2 + pd.tseries.offsets.DateOffset(years=-1)
        idx = self.zc_curves.index.intersection(pd.bdate_range(d1, d2))
        cov = np.cov(self.zc_curves.diff().loc[idx].T)
        if self.rebal_rule == 'vol':
            risk_dev = np.sqrt(261) * np.sqrt(np.dot(np.dot(tracking_error.to_numpy(), cov), tracking_error.to_numpy().T))
        elif self.rebal_rule == 'l2':
            risk_dev = np.sqrt(np.sum(tracking_error ** 2))
        if risk_dev > self.rebal_threshold:
            target_position = self.order_to_open(date, target_risk, spot_curve_handle)
            order = target_position.subtract(position, fill_value=0)
        else:
            order = pd.Series(index=position.index, data=0)
        return order
    
    def calc_position(self, min_lot=0):
        print('Calculating Position')
        if self.strategy.target_risk is None:
            self.strategy.estimate_target_risk()
        strategy_risk = self.strategy.target_risk
        target_risk = pd.DataFrame(columns = self.tenors, index = strategy_risk.index, data=0)
        target_risk.loc[strategy_risk.index, strategy_risk.columns] = strategy_risk
        df_position = pd.DataFrame(index=target_risk.index, columns=self.instrument_universe.index, data=0)
        df_risk = pd.DataFrame(index=target_risk.index, columns=target_risk.columns, data=0)
        df_trades = pd.DataFrame(index=target_risk.index, columns=self.instrument_universe.index, data=0)
        d = target_risk.index[0]
        spot_curve_handle = self.get_ql_obj(d)
        target_order = self.order_to_open(d, target_risk.loc[d], spot_curve_handle)
        df_trades.loc[d, target_order.index] = target_order * (target_order.abs() > min_lot)
        df_position.loc[d] = df_trades.loc[d]
        df_risk.loc[d] = self.estimate_risk(df_position.loc[d], d, spot_curve_handle)
        for i in tqdm(range(1, len(target_risk.index))):
            d = target_risk.index[i]
            d_signal = target_risk.index[i-self.time_slippage]
            spot_curve_handle = self.get_ql_obj(d)
            df_position.loc[d] = df_position.shift(1).loc[d]
            target_order = self.rebalance_order(df_position.loc[d], target_risk.loc[d_signal], d, spot_curve_handle)
            df_trades.loc[d, target_order.index] = target_order * (target_order.abs() > min_lot)
            df_position.loc[d] += df_trades.loc[d]
            current_risk = self.estimate_risk(df_position.loc[d], d, spot_curve_handle)
            df_risk.loc[d] = current_risk
        self.position = df_position
        self.risk = df_risk
        self.trades = df_trades
        return self.position, self.risk, self.trades
        
    def calc_total_pnl(self, tcosts=0):
        self.tcosts = tcosts
        repo_rates = self.zc_curves.loc[self.position.index, ['Y1']]
        index_name = self.position.index.name
        sett_dates = list(self.position.reset_index()[index_name].apply(ql.Date().from_date).map(
            lambda x: self.calendar.advance(x, self.settlement_days, ql.Days)))
        df_position_nonzero = self.position.loc[:, (self.position != 0).any(axis=0)].copy()
        df_settlement = pd.DataFrame(index=self.position.index, columns=['Settle Date'], data=sett_dates)
        df_settlement['Funding Days'] = df_settlement['Settle Date'].diff().shift(-1)
        df_sett_position = self.position.copy()
        df_sett_position.index = sett_dates
        repo_rates.index = sett_dates
        repo_rates = repo_rates.iloc[:,0]
        df_dirty_prices = pd.DataFrame(index=self.position.index, columns=self.market_data.columns, data=0)
        df_tcosts = pd.DataFrame(index=self.trades.index, columns=self.market_data.columns, data=0)
        df_coupons = pd.DataFrame(index=self.position.index, columns=self.position.columns, data=0)
        for b in df_position_nonzero.columns:
            bond = self.instrument_universe.loc[b, 'QL bond']
            coupon_dates = []
            coupon_amounts = []
            for c in bond.cashflows()[:-1]:
                coupon_dates.append(c.date())
                coupon_amounts.append(c.amount())
            for d in self.position.index:
                ql_d = ql.Date().from_date(d)
                ql_d_settle = df_settlement.loc[d, 'Settle Date']
                if ql_d_settle in coupon_dates:
                    df_coupons.loc[d, b] = coupon_amounts[coupon_dates.index(ql_d_settle)]
                df_dirty_prices.loc[d, b] = self.market_data.loc[d, b] + \
                    bond.accruedAmount(self.calendar.advance(ql_d, self.settlement_days, ql.Days))
                try:
                    df_tcosts.loc[d, b] = self.market_data.loc[d, b] - bond.cleanPrice(
                        bond.bondYield(self.market_data.loc[d, b], self.day_count, self.yield_basis, ql.Compounded) + self.tcosts/4,
                        self.day_count, self.yield_basis, ql.Compounded)
                except:
                    df_tcosts.loc[d, b] = 0
        self.coupons = df_coupons * self.position.shift(1) / 100
        df_sett_position.index = df_settlement.index
        repo_rates.index = df_settlement.index
        self.funding = -(df_sett_position * df_dirty_prices / 100).mul(
            repo_rates, axis=0).mul(
                df_settlement['Funding Days'], axis=0) / 360
        self.tcosts = -df_tcosts * self.trades.abs() / 100
        self.pnl = ((self.position.shift(1) * df_dirty_prices.diff() / 100)+self.funding+self.coupons + self.tcosts)
        self.dirty_prices = df_dirty_prices
        self.tcosts_all = df_tcosts
        return self.pnl.sum(axis=1)

def trading_order(instruments, date, risk_to_trade, spot_curve_handle, curve_tenors_years,
                  fit_data=None, fit_threshold=0.01, maturity_band=0.2):
    ql_date = ql.Date(date.day, date.month, date.year)
    ql.Settings.instance().evaluationDate = ql_date
    curve_tenors = ['Y'+str(t) for t in curve_tenors_years]
    sensitivities = instruments[instruments['PricingDate'] < date].copy()
    for i in sensitivities.index:
        bond = sensitivities.loc[i, 'QL bond']
        sensitivities.loc[i, curve_tenors] = bond_sensitivities(bond, date, spot_curve_handle, curve_tenors_years)
    risk = pd.Series(index= curve_tenors, data=0)
    risk.loc[risk_to_trade.index] = risk_to_trade
    if fit_data is not None:
        benchmarks = []
        to_keep = fit_data[fit_data.abs() <= fit_threshold].index
        sensitivities = sensitivities.loc[to_keep.intersection(sensitivities.index)]
        for t in curve_tenors:
            years = int(t[1:])
            tenor_date = date + pd.tseries.offsets.DateOffset(years=years)
            mask = (abs((sensitivities['Maturity'] - tenor_date).dt.days) < maturity_band*years*365)
            subuniverse = sensitivities.loc[mask]
            candidates = list(subuniverse.index)
            if len(candidates) == 0:
                benchmarks.append(sensitivities[t].abs().idxmax())
            else:
                benchmarks.append((np.sign(risk_to_trade[t]) * fit_data[candidates]).idxmin())
        benchmarks = list(set(benchmarks))
    else:
        benchmarks = list(set([sensitivities[t].abs().idxmax() for t in curve_tenors]))
    order = pd.Series(index = benchmarks,
                    data=100 * np.linalg.lstsq(sensitivities.loc[benchmarks, curve_tenors].to_numpy().T, risk.to_numpy(), rcond=None)[0])
    return order

def portfolio_risk(instruments, date, portfolio, spot_curve_handle, tenors_years):
    risk = np.array([0.0] * len(tenors_years))
    for i in portfolio.index:
        bond = instruments.loc[i, 'QL bond']
        risk += bond_sensitivities(bond, date, spot_curve_handle, tenors_years) * portfolio.loc[i] / 100
    return risk

def bond_sensitivities(bond, date, spot_curve_handle, tenors_years):
    ql_date = ql.Date(date.day, date.month, date.year)
    ql.Settings.instance().evaluationDate = ql_date
    bumps = [ql.SimpleQuote(0.00) for n in ([0] + tenors_years)]
    spreads = [ql.QuoteHandle(bump) for bump in bumps]
    spot_dates = [ql_date + ql.Period(int(t), ql.Years) for t in ([0] + tenors_years)]
    spreaded_yts = ql.YieldTermStructureHandle(
        ql.SpreadedLinearZeroInterpolatedTermStructure(spot_curve_handle, spreads, spot_dates))
    spreaded_yts.enableExtrapolation()
    bond.setPricingEngine(ql.DiscountingBondEngine(spreaded_yts))
    price = bond.cleanPrice()
    senstivities = []
    for bump in bumps[1:]:
        bump.setValue(0.0001)
        senstivities.append(bond.cleanPrice() - price)
        bump.setValue(0.00)
    return np.array(senstivities)

def trading_order(instruments, date, risk_to_trade, spot_curve_handle, curve_tenors_years,
                  fit_data=None, fit_threshold=0.01, maturity_band=0.2):
    ql_date = ql.Date(date.day, date.month, date.year)
    ql.Settings.instance().evaluationDate = ql_date
    curve_tenors = ['Y'+str(t) for t in curve_tenors_years]
    sensitivities = instruments[instruments['PricingDate'] < date].copy()
    for i in sensitivities.index:
        bond = sensitivities.loc[i, 'QL bond']
        sensitivities.loc[i, curve_tenors] = bond_sensitivities(bond, date, spot_curve_handle, curve_tenors_years)
    risk = pd.Series(index= curve_tenors, data=0)
    risk.loc[risk_to_trade.index] = risk_to_trade
    if fit_data is not None:
        benchmarks = []
        to_keep = fit_data[fit_data.abs() <= fit_threshold].index
        sensitivities = sensitivities.loc[to_keep.intersection(sensitivities.index)]
        for t in curve_tenors:
            years = int(t[1:])
            tenor_date = date + pd.tseries.offsets.DateOffset(years=years)
            mask = (abs((sensitivities['Maturity'] - tenor_date).dt.days) < maturity_band*years*365)
            subuniverse = sensitivities.loc[mask]
            candidates = list(subuniverse.index)
            if len(candidates) == 0:
                benchmarks.append(sensitivities[t].abs().idxmax())
            else:
                benchmarks.append((np.sign(risk_to_trade[t]) * fit_data[candidates]).idxmin())
        benchmarks = list(set(benchmarks))
    else:
        benchmarks = list(set([sensitivities[t].abs().idxmax() for t in curve_tenors]))
    order = pd.Series(index = benchmarks,
                    data=100 * np.linalg.lstsq(sensitivities.loc[benchmarks, curve_tenors].to_numpy().T, risk.to_numpy(), rcond=None)[0])
    return order

def portfolio_risk(instruments, date, portfolio, spot_curve_handle, tenors_years):
    risk = np.array([0.0] * len(tenors_years))
    for i in portfolio.index:
        bond = instruments.loc[i, 'QL bond']
        risk += bond_sensitivities(bond, date, spot_curve_handle, tenors_years) * portfolio.loc[i] / 100
    return risk

def bond_sensitivities(bond, date, spot_curve_handle, tenors_years):
    ql_date = ql.Date(date.day, date.month, date.year)
    ql.Settings.instance().evaluationDate = ql_date
    bumps = [ql.SimpleQuote(0.00) for n in ([0] + tenors_years)]
    spreads = [ql.QuoteHandle(bump) for bump in bumps]
    spot_dates = [ql_date + ql.Period(int(t), ql.Years) for t in ([0] + tenors_years)]
    spreaded_yts = ql.YieldTermStructureHandle(
        ql.SpreadedLinearZeroInterpolatedTermStructure(spot_curve_handle, spreads, spot_dates))
    spreaded_yts.enableExtrapolation()
    bond.setPricingEngine(ql.DiscountingBondEngine(spreaded_yts))
    price = bond.cleanPrice()
    senstivities = []
    for bump in bumps[1:]:
        bump.setValue(0.0001)
        senstivities.append(bond.cleanPrice() - price)
        bump.setValue(0.00)
    return np.array(senstivities)

