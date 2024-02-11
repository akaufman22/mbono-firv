#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 14:37:17 2022

@author: alex
"""

import pandas as pd
import numpy as np
import QuantLib as ql
import sqlite3

import dash
from dash import html, dash_table, dcc
from dash.dash_table.Format import Format, Scheme
import plotly.graph_objs as go
from statsmodels.tsa.stattools import adfuller

from MBONOCurveFit import *
from models import *
from ts_stat import *

# Constants###
###############
TENORS = [1, 2, 3, 5, 10, 20, 30]
BELLY = [2, 3, 5, 10, 20]
SE = [2, 3, 5]
LE = [5, 10, 20]
VALDATE = ql.Date(21, 11, 2022)  # should be ql.Date.todaysDate() when live

# Loading Data and calculating metrics###
##########################################
input_data = pd.read_csv("../../liveprices/MBONOsnapshot.csv", index_col=0)

model_belly, model_se, model_le = get_updated_models()
models = pd.DataFrame(index=['Belly', 'SE', 'LE'], columns=['Model',
                                                            'Tenors',
                                                            'TenorsStr',
                                                            'PC1', 'PC2',
                                                            'PC3'])
models['Model'] = [model_belly, model_se, model_le]
models['Tenors'] = [BELLY, SE, LE]
models['TenorsStr'] = [[str(i)+'Y' for i in t] for t in list(models['Tenors'])]

LiveFit = MBONOCurveFit(input_data, date=VALDATE)
zero_curve = LiveFit.get_zero_curve(TENORS)['Rate']
fit_table = LiveFit.fit.sort_values(['Maturity'])[['Name', 'Maturity',
                                                   'Price', 'Yield',
                                                   'Th Price', 'Th Yield',
                                                   'Difference']].copy()
fit_table['Z-score'] = get_fit_zscores(LiveFit.fit['Difference'])
fit_table['Difference'] = 10000 * fit_table['Difference']
curve_table = pd.DataFrame(index=['Rate', 'Belly PC1', 'Belly PC2',
                                  'SE PC1', 'SE PC2', 'LE PC1', 'LE PC2'],
                           columns=zero_curve.index)
curve_table.loc['Rate'] = zero_curve
for m in models.index:
    model = models.loc[m, 'Model']
    cols = models.loc[m, 'TenorsStr']
    curve_table.loc[m + ' PC1', cols] = model_residuals(model,
                                                        zero_curve[cols], 1)
    curve_table.loc[m + ' PC2', cols] = model_residuals(model,
                                                        zero_curve[cols], 2)
    models.loc[m, ['PC1', 'PC2', 'PC3']] = np.dot(
        zero_curve[cols].values - model.mean_, model.components_.T)

con = sqlite3.connect("../../db/MBONOdata.db")
query = """SELECT Date, BellyPC1, BellyPC2, BellyPC3, SEPC1, SEPC2, SEPC3,
LEPC1, LEPC2, LEPC3
FROM PCLoadings ORDER BY Date DESC LIMIT 2600;"""
pc_series = pd.read_sql_query(query, con, parse_dates=['Date'],
                                index_col='Date')

components_stats = pd.DataFrame(index=[m + ' ' + c for m in models.index
                                       for c in ['PC1', 'PC2', 'PC3']],
                                columns=['Zsc', 'ADF p-val', 'HE', 'HL'])
for c in ['PC1', 'PC2', 'PC3']:
    for m in models.index:
        time_series = pc_series[m + c]
        components_stats.loc[m + ' ' + c, 'Zsc'] = \
            (models.loc[m, c] - time_series.mean()) / time_series.std()
        components_stats.loc[m + ' ' + c, 'ADF p-val'] = adfuller(
            time_series)[1]
        components_stats.loc[m + ' ' + c, 'HE'] = \
            hurst_exponent(time_series.values)
        components_stats.loc[m + ' ' + c, 'HL'] = HL(time_series.values)

# Configuring Charts for Dashboard###
######################################
fig_pc1 = go.Figure()
fig_pc1.add_trace(
    go.Scatter(name='Belly',
               x=pc_series.index,
               y=pc_series['BellyPC1'],
               line=dict(color='blue')))
fig_pc1.add_trace(
    go.Scatter(name='Belly live',
               x=pc_series.index,
               y=[models.loc['Belly', 'PC1']] * len(pc_series.index),
               line=dict(color='lightblue')))
fig_pc1.add_trace(
    go.Scatter(name='Short End',
               x=pc_series.index,
               y=pc_series['SEPC1'],
               line=dict(color='red')))
fig_pc1.add_trace(
    go.Scatter(name='Short End live',
               x=pc_series.index,
               y=[models.loc['SE', 'PC1']] * len(pc_series.index),
               line=dict(color='pink')))
fig_pc1.add_trace(
    go.Scatter(name='Long End',
               x=pc_series.index,
               y=pc_series['LEPC1'],
               line=dict(color='green')))
fig_pc1.add_trace(
    go.Scatter(name='Lng End  live',
               x=pc_series.index,
               y=[models.loc['LE', 'PC1']] * len(pc_series.index),
               line=dict(color='lightgreen')))
fig_pc1.update_layout(
    title_text='1st PC',
    updatemenus=[dict(
        type='buttons',
        direction='right',
        active=0,
        x=1,
        y=1.2,
        buttons=list([
            dict(label='All',
                 method='update',
                 args=[{'visible': [True] * 6},
                       {'title': '1st PC'}]),
            dict(label='Belly',
                 method='update',
                 args=[{'visible': [True] * 2 + [False] * 4},
                       {'title': '1st PC Belly'}]),
            dict(label='SE',
                 method='update',
                 args=[{'visible': [False] * 2 + [True] * 2 + [False] * 2},
                       {'title': '1st PC Short End'}]),
            dict(label='LE',
                 method='update',
                 args=[{'visible': [False] * 4 + [True] * 2},
                       {'title': '1st PC Long End'}])]))],
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label='1m',
                     step='month',
                     stepmode='backward'),
                dict(count=6,
                     label='6m',
                     step='month',
                     stepmode='backward'),
                dict(count=1,
                     label='YTD',
                     step='year',
                     stepmode='todate'),
                dict(count=1,
                     label='1y',
                     step='year',
                     stepmode='backward'),
                dict(step='all')
                ])
            ),
        rangeslider=dict(visible=True),
        type='date'
        ),
    xaxis_title='Date',
    yaxis_title='Factor value',
    margin=dict(l=75, r=5, b=50, t=35))

fig_pc2 = go.Figure()
fig_pc2.add_trace(
    go.Scatter(name='Belly',
               x=pc_series.index,
               y=pc_series['BellyPC2'],
               line=dict(color='blue')))
fig_pc2.add_trace(
    go.Scatter(name='Belly live',
               x=pc_series.index,
               y=[models.loc['Belly', 'PC2']] * len(pc_series.index),
               line=dict(color='lightblue')))
fig_pc2.add_trace(
    go.Scatter(name='Short End',
               x=pc_series.index,
               y=pc_series['SEPC2'],
               line=dict(color='red')))
fig_pc2.add_trace(
    go.Scatter(name='Short End live',
               x=pc_series.index,
               y=[models.loc['SE', 'PC2']] * len(pc_series.index),
               line=dict(color='pink')))
fig_pc2.add_trace(
    go.Scatter(name='Long End',
               x=pc_series.index,
               y=pc_series['LEPC2'],
               line=dict(color='green')))
fig_pc2.add_trace(
    go.Scatter(name='Lng End  live',
               x=pc_series.index,
               y=[models.loc['LE', 'PC2']] * len(pc_series.index),
               line=dict(color='lightgreen')))
fig_pc2.update_layout(
    title_text='2nd PC',
    updatemenus=[dict(
        type='buttons',
        direction='right',
        active=0,
        x=1,
        y=1.2,
        buttons=list([
            dict(label='All',
                 method='update',
                 args=[{'visible': [True] * 6},
                       {'title': '2nd PC'}]),
            dict(label='Belly',
                 method='update',
                 args=[{'visible': [True] * 2 + [False] * 4},
                       {'title': '2nd PC Belly'}]),
            dict(label='SE',
                 method='update',
                 args=[{'visible': [False] * 2 + [True] * 2 + [False] * 2},
                       {'title': '2nd PC Short End'}]),
            dict(label='LE',
                 method='update',
                 args=[{'visible': [False] * 4 + [True] * 2},
                       {'title': '2nd PC Long End'}])]))],
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label='1m',
                     step='month',
                     stepmode='backward'),
                dict(count=6,
                     label='6m',
                     step='month',
                     stepmode='backward'),
                dict(count=1,
                     label='YTD',
                     step='year',
                     stepmode='todate'),
                dict(count=1,
                     label='1y',
                     step='year',
                     stepmode='backward'),
                dict(step='all')
                ])
            ),
        rangeslider=dict(visible=True),
        type='date'
        ),
    xaxis_title='Date',
    yaxis_title='Factor value',
    margin=dict(l=75, r=5, b=50, t=35))

fig_pc3 = go.Figure()
fig_pc3.add_trace(
    go.Scatter(name='Belly',
               x=pc_series.index,
               y=pc_series['BellyPC3'],
               line=dict(color='blue')))
fig_pc3.add_trace(
    go.Scatter(name='Belly live',
               x=pc_series.index,
               y=[models.loc['Belly', 'PC3']] * len(pc_series.index),
               line=dict(color='lightblue')))
fig_pc3.add_trace(
    go.Scatter(name='Short End',
               x=pc_series.index,
               y=pc_series['SEPC3'],
               line=dict(color='red')))
fig_pc3.add_trace(
    go.Scatter(name='Short End live',
               x=pc_series.index,
               y=[models.loc['SE', 'PC3']] * len(pc_series.index),
               line=dict(color='pink')))
fig_pc3.add_trace(
    go.Scatter(name='Long End',
               x=pc_series.index,
               y=pc_series['LEPC3'],
               line=dict(color='green')))
fig_pc3.add_trace(
    go.Scatter(name='Lng End  live',
               x=pc_series.index,
               y=[models.loc['LE', 'PC3']] * len(pc_series.index),
               line=dict(color='lightgreen')))
fig_pc3.update_layout(
    title_text='3rd PC',
    updatemenus=[dict(
        type='buttons',
        direction='right',
        active=0,
        x=1,
        y=1.2,
        buttons=list([
            dict(label='All',
                 method='update',
                 args=[{'visible': [True] * 6},
                       {'title': '3rd PC'}]),
            dict(label='Belly',
                 method='update',
                 args=[{'visible': [True] * 2 + [False] * 4},
                       {'title': '3rd PC Belly'}]),
            dict(label='SE',
                 method='update',
                 args=[{'visible': [False] * 2 + [True] * 2 + [False] * 2},
                       {'title': '3rd PC Short End'}]),
            dict(label='LE',
                 method='update',
                 args=[{'visible': [False] * 4 + [True] * 2},
                       {'title': '3rd PC Long End'}])]))],
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label='1m',
                     step='month',
                     stepmode='backward'),
                dict(count=6,
                     label='6m',
                     step='month',
                     stepmode='backward'),
                dict(count=1,
                     label='YTD',
                     step='year',
                     stepmode='todate'),
                dict(count=1,
                     label='1y',
                     step='year',
                     stepmode='backward'),
                dict(step='all')
                ])
            ),
        rangeslider=dict(visible=True),
        type='date'
        ),
    xaxis_title='Date',
    yaxis_title='Factor value',
    margin=dict(l=75, r=5, b=50, t=35))

# Confiduring tables for Dashboard###
######################################
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
    dict(id='Difference', name='Spread, bp', type='numeric',
         format=Format(precision=0, scheme=Scheme.fixed)),
    dict(id='Z-score', name='Spread Zsc', type='numeric',
         format=Format(precision=2, scheme=Scheme.fixed))
    ]

stats_settings = [
    dict(id='index', name='', type='text'),
    dict(id='Zsc', name='Z-score', type='numeric',
         format=Format(precision=2, scheme=Scheme.fixed)),
    dict(id='ADF p-val', name='ADF p-value', type='numeric',
         format=Format(precision=2, scheme=Scheme.fixed)),
    dict(id='HE', name='Hurst', type='numeric',
         format=Format(precision=2, scheme=Scheme.fixed)),
    dict(id='HL', name='Half life', type='numeric',
         format=Format(precision=0, scheme=Scheme.fixed))
    ]

# Creating Layout###
#####################
app = dash.Dash()

app.layout = html.Div(children=[
    html.H1(children='MBONO Market'),
    html.Div([html.H4('Individual Issues Rich/Cheap'),
              dash_table.DataTable(id='live-market-table',
                                   columns=column_settings,
                                   data=fit_table.to_dict('rows'),
                                   style_data_conditional=[{'if': {
                                       'filter_query': '{Difference} > 4.5',
                                       'column_id': 'Difference'},
                                       'backgroundColor': 'green',
                                       'color': 'white'}, {
                                           'if': {
                                               'filter_query':
                                                   '{Difference} < -4.5',
                                                   'column_id': 'Difference'
                                                   },
                                               'backgroundColor': 'red',
                                               'color': 'white'
                                               },],)],
             style={
                                               'width': '49%',
                                               'display': 'inline-block',
                                               'vertical-align': 'top',
                                               'margin-right': '1%'}),
    html.Div([dcc.Tabs([dcc.Tab(label='Yield Curve', children=[
        html.Div([
            html.H4('Market vs Theoretical Yield Curves'),
            dcc.Graph(
                id='yield-curve',
                figure={
                    'data': [go.Scatter(x=fit_table['Maturity'],
                                        y=fit_table['Yield'],
                                        name='Market', text=fit_table['Name']),
                             go.Scatter(x=fit_table['Maturity'],
                                        y=fit_table['Th Yield'],
                                        name='Theoretical',
                                        text=fit_table['Name'])],
                    'layout': go.Layout(
                        yaxis=dict(tickformat='.2%', title='Yield'),
                        xaxis=dict(title='Maturity'),
                        legend=dict(orientation='h'),
                        margin=dict(l=75, r=5, b=50, t=35),
                        height=300)})]),
        html.Div([
            html.H4('Spot Zero Coupon Curve Rich/Cheap'),
            dash_table.DataTable(
                id='zero-curve',
                columns=[dict(id='index', name='', type='text')] +
                [dict(id=i, name=i, type='numeric',
                      format=Format(precision=2,
                                    scheme=Scheme.percentage))
                 for i in curve_table.columns],
                data=curve_table.reset_index().to_dict('rows'))])
        ]), dcc.Tab(label='Charts', children=[
                dcc.Graph(id='1st', figure=fig_pc1),
                dcc.Graph(id='2nd', figure=fig_pc2),
                dcc.Graph(id='3rd', figure=fig_pc3)]),
            dcc.Tab(label='Stats', children=[
                 html.Div([
                     html.H4('PCA Stats'),
                     dash_table.DataTable(
                         id='stats',
                         columns=stats_settings,
                         data=components_stats.reset_index().to_dict('rows'),
                         style_data_conditional=[
                             {'if': {
                                 'filter_query': '{Zsc} > 1.5',
                                 'column_id': 'Zsc'},
                                 'backgroundColor': 'green',
                                 'color': 'white'},
                             {'if': {
                                 'filter_query': '{Zsc} < -1.5',
                                 'column_id': 'Zsc'},
                                 'backgroundColor': 'red',
                                 'color': 'white'},
                             {'if': {
                                 'filter_query': '{ADF p-val} <= 0.05',
                                 'column_id': 'ADF p-val'},
                                 'backgroundColor': '#7FDBFF',
                                 'color': 'white'},
                             {'if': {
                                 'filter_query': '{HE} <= 0.4',
                                 'column_id': 'HE'},
                                 'backgroundColor': '#7FDBFF',
                                 'color': 'white'},
                             {'if': {
                                 'filter_query': '{HL} <= 65',
                                 'column_id': 'HL'},
                                 'backgroundColor': '#7FDBFF',
                                 'color': 'white'}
                             ])])])
            ])],
        style={'width': '49%',
               'display': 'inline-block',
               'vertical-align': 'top',
               'margin-top': '1%'})])
if __name__ == '__main__':
    app.run_server()
