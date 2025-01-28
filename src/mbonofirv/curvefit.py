#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 15:14:11 2022

@author: alex
"""
import QuantLib as ql

from curvemath import *


class MBONOCurveFit:
    """
    Class to fit a curve to a set of bonds and provide a valuation
    """

    def __init__(
        self,
        market_data,
        date=ql.Date.todaysDate(),
        id_type="BBGID",
        min_days=270,
        outlier_range=0.001,
    ):
        self.date = date
        self.market_data = market_data
        self.id_type = id_type
        self.min_days = min_days
        self.outlier_range = outlier_range

        self.curve, self.fit = clean_fit(
            mbono_market_snapshot(market_data, valuation_date=date, id_type=id_type),
            valuation_date=date,
            min_days=min_days,
            outlier_range=outlier_range,
        )

    def get_val_data(self, market_data, id_type):
        val_data = evaluate_bonds(
            mbono_market_snapshot(
                market_data, valuation_date=self.date, id_type=self.id_type
            ),
            self.curve,
            valuation_date=self.date,
        )
        return val_data

    def get_zero_curve(self, tenors):
        zero_rates = zero_df_curve(self.curve, tenors)
        return zero_rates
