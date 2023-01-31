#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 16:06:39 2023

@author: alex
"""
import pandas as pd
import numpy as np
import sqlite3
import statsmodels.api as sm


def hurst_exponent(data):
    
    lags = range(2,20)
    tau = [np.sqrt(np.std(np.subtract(data[lag:], data[:-lag]))) for lag in lags]
    m = np.polyfit(np.log(lags), np.log(tau), 1)
    hurst = m[0]*2
    return hurst

def HL(time_series):
    inc = np.subtract(time_series[1:], time_series[:-1])
    model = sm.OLS(inc, sm.add_constant(time_series[:-1]))
    res = model.fit()
    HL = -np.log(2)/res.params[1]
    return HL

def get_fit_zscores(fit_data, days = 60, id_type = 'BBGID'):
    con = sqlite3.connect("../../db/MBONOdata.db")
    str_ids = '('+str(list(fit_data.index))[1:-1]+')'
    limit = days * len(fit_data.index)
    query = """SELECT Instruments.%s, FitData.Date, 
    FitData.Residual
    FROM FitData INNER JOIN Instruments
    ON FitData.InstrumentID = Instruments.InstrumentID
    WHERE Instruments.%s in %s ORDER BY FitData.Date DESC LIMIT %d;
    """ %(id_type, id_type, str_ids, limit)
    aggregate_residuals = pd.read_sql_query(query, con, parse_dates = ['Date'])
    zscores = pd.Series(index = fit_data.index, dtype = 'float64')
    for i in fit_data.index:
        sample = aggregate_residuals[
            aggregate_residuals[id_type] == i]['Residual']
        zscores[i] = (fit_data[i] - sample.mean()) / sample.std()
    return zscores