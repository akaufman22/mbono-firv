#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 14:37:17 2022

@author: alex
"""

import pandas as pd
import QuantLib as ql
import sqlite3


import dash
from dash import html, dash_table, dcc
from dash.html import Div
from dash.dash_table import DataTable, FormatTemplate
from dash.dash_table.Format import Format, Scheme
import plotly.graph_objs as go

from MBONOCurveFit import *

input_data = pd.read_csv("../../liveprices/MBONOsnapshot.csv", index_col = 0)

LiveFit = MBONOCurveFit(input_data)
fit_table = LiveFit.fit.sort_values(['Maturity'])[['Name', 'Maturity', 
                                                   'Price', 'Yield', 
                                                   'Th Price', 'Th Yield',
                                                   'Difference']].copy()


app = dash.Dash()
column_settings = [
    dict(id='Name', name='Name', type='text'),
    dict(id='Price', name='Price', type='numeric', 
         format=Format(precision=2, scheme=Scheme.fixed)),
    dict(id='Th Price', name='Th Price', type='numeric', 
         format=Format(precision=2, scheme=Scheme.fixed)),
    dict(id='Yield', name='Yield', type='numeric', 
         format=Format(precision=2, scheme=Scheme.percentage)),
    dict(id='Th Yield', name='Th Yield', type='numeric', 
         format=Format(precision=2, scheme=Scheme.percentage)),
    dict(id='Difference', name='Spread to Curve', type='numeric', 
         format=Format(precision=4, scheme=Scheme.percentage))
    ]
app.layout = html.Div(children = [
    html.H1(children = 'MBONO Market'),
    html.Div([dash_table.DataTable(id = 'live-market-table',
                                   columns = column_settings,
                                   data = fit_table.to_dict('rows'))], 
             style = {
                                               'width':'49%',
                                               'display':'inline-block',
                                               'vertical-align':'top',
                                               'margin-right':'1%'}),
    html.Div([dcc.Graph(
        id = 'yield-curve',
        figure = {
            'data':[go.Scatter(x=fit_table['Maturity'], y=fit_table['Yield'],
                               name='Market', text=fit_table['Name']),
                    go.Scatter(x=fit_table['Maturity'], y=fit_table['Th Yield'],
                               name='Theoretical', text=fit_table['Name'])],
            'layout': go.Layout(
                yaxis=dict(tickformat='.2%', title='Yield'),
                xaxis=dict(title='Maturity'),
                legend = dict(),
                height=400)})],
        style = {'width':'49%',
                 'display':'inline-block',
                 'vertical-align':'top',
                 'margin-top':'1%'})])
if __name__ == '__main__':
    app.run_server()