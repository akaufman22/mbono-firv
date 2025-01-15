#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 12:09:14 2022

@author: alex
"""

import pandas as pd
import numpy as np
import sqlite3
from sklearn.decomposition import PCA
import pickle


def get_latest_curve_data():
    con = sqlite3.connect("../../db/MBONOdata.db")
    query = """SELECT Date, Y2, Y3, Y5, Y10, Y20
    FROM ZeroCurves ORDER BY Date DESC LIMIT 2600;"""
    curve_data = pd.read_sql_query(query, con, parse_dates=['Date'],
                                   index_col=['Date']).sort_index()
    return curve_data


def models_updated():
    try:
        con = sqlite3.connect("../../db/MBONOdata.db")
        cur = con.cursor()
        res = cur.execute('SELECT MAX(Date) FROM ZeroCurves;')
        curve_latest = res.fetchone()[0]
        res = cur.execute('SELECT MAX(Date) FROM PCLoadings;')
        model_latest = res.fetchone()[0]
        return (curve_latest == model_latest)
    except:
        return False


def update_models():
    path = "../../models/"
    curve_data = get_latest_curve_data()
    se = ['Y2', 'Y3', 'Y5']
    le = ['Y5', 'Y10', 'Y20']

    model_belly = PCA(n_components=3)
    model_belly.fit(curve_data)
    belly_loadings = model_belly.fit_transform(curve_data)
    pickle.dump(model_belly, open(path + 'belly.pkl', 'wb'))

    model_se = PCA(n_components=3)
    model_se.fit(curve_data[se])
    se_loadings = model_se.fit_transform(curve_data[se])
    pickle.dump(model_se, open(path + 'se.pkl', 'wb'))

    model_le = PCA(n_components=3)
    model_le.fit(curve_data[le])
    le_loadings = model_le.fit_transform(curve_data[le])
    pickle.dump(model_le, open(path + 'le.pkl', 'wb'))

    con = sqlite3.connect("../../db/MBONOdata.db")
    cur = con.cursor()
    cur.execute("DELETE FROM PCLoadings;")
    query = """INSERT INTO PCLoadings  (Date, BellyPC1, BellyPC2, BellyPC3,
        SEPC1, SEPC2, SEPC3, LEPC1, LEPC2, LEPC3)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);"""

    for i, d in enumerate(curve_data.index):
        vector = [d.strftime('%Y-%m-%d'), belly_loadings[i,0],
                  belly_loadings[i, 1], belly_loadings[i, 2],
                  se_loadings[i, 0], se_loadings[i, 1], se_loadings[i, 2],
                  le_loadings[i, 0], le_loadings[i, 1], le_loadings[i, 2]]
        cur.executemany(query, (vector,))
        con.commit()

    return model_belly, model_se, model_le


def get_updated_models():
    if models_updated():
        path = "../../models/"
        model_belly = pickle.load(open(path + 'belly.pkl', 'rb'))
        model_se = pickle.load(open(path + 'se.pkl', 'rb'))
        model_le = pickle.load(open(path + 'le.pkl', 'rb'))
        return model_belly, model_se, model_le
    else:
        return update_models()


def model_residuals(model, vector, k):
    dimensions = len(model.components_[0])
    full_w = model.components_
    reduced_w = np.append(full_w[:k], np.zeros([dimensions - k, dimensions]))\
        .reshape(dimensions, dimensions)
    residuals = vector - model.mean_ - np.dot(np.dot(vector - model.mean_,
                                                     reduced_w.T), reduced_w)
    return residuals
