#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 09:13:27 2022

@author: alex
"""

from MBONOCurveFit import *

import pandas as pd
import QuantLib as ql
import sqlite3

#### Global curve tenors in years
TENORS = [1,2,3,5,10,20,30]
####


def load_agg_price_data():
    # Loading portion of raw price data without yet calculated curves
    con = sqlite3.connect("../../db/MBONOdata.db")
    query = """SELECT InstrumentID,  Date, Price FROM MarketData 
    WHERE Date NOT IN (SELECT Date FROM ZeroCurves);"""
    agg_price_data = pd.read_sql_query(query, con, parse_dates=['Date']). \
        sort_values(['Date'])
    return agg_price_data


def update_tables():
    # Procedure updates tables with zero curves and with goodness of fit
    full_data = load_agg_price_data()
    dates_to_update = full_data['Date'].unique()
    con = sqlite3.connect("../../db/MBONOdata.db")
    cur = con.cursor()
    # Retreiving current IDs for a counter
    res = cur.execute('SELECT MAX(ZeroCurveID) FROM ZeroCurves;')
    query_output = res.fetchone()
    if query_output[0] is None:
        CurveID = 0
    else:
        CurveID = query_output[0]
    res = cur.execute('SELECT MAX(DataPointID) FROM FitData;')
    query_output = res.fetchone()
    if query_output[0] is None:
        DataPointID = 0
    else:
        DataPointID = query_output[0]
    # Updating the tables iteratively for each date
    for d in dates_to_update:
        date = ql.Date().from_date(pd.Timestamp(d))
        date_str = pd.Timestamp(d).strftime('%Y-%m-%d')
        price_snapshot = full_data[full_data['Date'] ==
                                   d][['InstrumentID', 'Price']]\
            .set_index(['InstrumentID']).copy()
        HistFit = MBONOCurveFit(price_snapshot, date, id_type='InstrumentID')
        residuals = HistFit.fit['Difference'].copy()
        zero_curve = HistFit.get_zero_curve(TENORS)['Rate'].copy()
        CurveID += 1
        query = """INSERT INTO ZeroCurves  (ZeroCurveID, Date, Y1, Y2, Y3, Y5,
        Y10, Y20, Y30) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);"""
        vector = [CurveID, date_str] + list(zero_curve)
        cur.executemany(query, (vector,))
        con.commit()
        for InstrumentID in residuals.index:
            DataPointID += 1
            query = """INSERT INTO FitData (DataPointID, InstrumentID,
            ZeroCurveID, Date, Residual) VALUES (?, ?, ?, ?, ?)"""
            vector = [DataPointID, InstrumentID, CurveID, date_str,
                      residuals[InstrumentID]]
            cur.executemany(query, (vector,))
            con.commit()
        print(date_str)
        print(HistFit.fit)
    return
