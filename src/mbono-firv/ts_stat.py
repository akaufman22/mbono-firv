#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 16:06:39 2023

@author: alex
"""
import pandas as pd
import numpy as np
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