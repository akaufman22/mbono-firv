#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 16:08:30 2022

@author: alex
"""

import pandas as pd
import QuantLib as ql
import sqlite3

# GLOBALS FOR MEX MARKET
CONVENTION = ql.Unadjusted
CALENDAR = ql.Mexico()
YIELD_BASIS = ql.Semiannual
TENOR = ql.Period(ql.Semiannual)
RULE = ql.DateGeneration.Backward
DAY_COUNT = ql.Actual360()
SETTLEMENT_DAYS = 2
###


def define_ql_bonds(instrument_ids, id_type="BBGID"):
    con = sqlite3.connect("../../db/MBONOdata.db")
    query = """SELECT Date, Coupon, Principal
    FROM StaticSchedules WHERE InstrumentID in
    (SELECT InstrumentID FROM Instruments WHERE BBGID = ?);"""
    bonds_sample = []
    for i in instrument_ids:
        bond_schedule = pd.read_sql_query(query, con, parse_dates=["Date"], params=(i,))
        bond_schedule["Days"] = (
            bond_schedule["Date"] - bond_schedule["Date"].shift(1)
        ).dt.days
        schedule = ql.Schedule(
            bond_schedule["Date"].apply(ql.Date().from_date).values,
            CALENDAR,
            CONVENTION,
            CONVENTION,
            TENOR,
            RULE,
            False,
        )
        face_value = sum(bond_schedule["Principal"])
        coupons = (
            ((360 / bond_schedule["Days"]) * bond_schedule["Coupon"] / face_value)
            .dropna()
            .values
        )
        bonds_sample.append(
            ql.FixedRateBond(SETTLEMENT_DAYS, 100, schedule, coupons, DAY_COUNT)
        )
    return bonds_sample


def define_ql_bonds_bulk(
    instrument_ids, id_type="BBGID", db_path="../../db/MBONOdata.db"
):
    """
    Define a list of QuantLib FixedRateBond objects from a list of instrument ids
    and a database path. The database should contain the tables Instruments and
    StaticSchedules with the corresponding fields."""
    con = sqlite3.connect(db_path)
    str_ids = "(" + str(instrument_ids)[1:-1] + ")"
    bonds_sample = []
    query = """SELECT Instruments.%s, StaticSchedules.Date,
    StaticSchedules.Coupon, StaticSchedules.Principal
    FROM StaticSchedules INNER JOIN Instruments
    ON StaticSchedules.InstrumentID = Instruments.InstrumentID
    WHERE Instruments.%s in %s;""" % (
        id_type,
        id_type,
        str_ids,
    )
    aggregate_schedules = pd.read_sql_query(query, con, parse_dates=["Date"])
    for b in instrument_ids:
        bond_schedule = (
            aggregate_schedules[aggregate_schedules[id_type] == b]
            .copy()
            .sort_values("Date")
        )
        bond_schedule["Days"] = (
            bond_schedule["Date"] - bond_schedule["Date"].shift(1)
        ).dt.days
        schedule = ql.Schedule(
            bond_schedule["Date"].apply(ql.Date().from_date).values,
            CALENDAR,
            CONVENTION,
            CONVENTION,
            TENOR,
            RULE,
            False,
        )
        face_value = sum(bond_schedule["Principal"])
        coupons = (
            ((360 / bond_schedule["Days"]) * bond_schedule["Coupon"] / face_value)
            .dropna()
            .values
        )
        bonds_sample.append(
            ql.FixedRateBond(SETTLEMENT_DAYS, 100, schedule, coupons, DAY_COUNT)
        )
    return bonds_sample


def fitted_ql_curve(
    bonds_sample,
    prices,
    valuation_date=ql.Date.todaysDate(),
    curve_fitting=ql.SvenssonFitting(),
):
    """Fit a QuantLib curve to a set of bonds and prices. The curve fitting
    method is specified by the curve_fitting argument. The valuation date is
    used to calculate the settlement date of the bonds."""
    settlement_date = CALENDAR.advance(
        valuation_date, ql.Period(SETTLEMENT_DAYS, ql.Days)
    )
    bond_helpers = []
    for i, b in enumerate(bonds_sample):
        bond_helpers.append(ql.BondHelper(ql.QuoteHandle(ql.SimpleQuote(prices[i])), b))
    yield_curve_fit = ql.FittedBondDiscountCurve(
        settlement_date, bond_helpers, DAY_COUNT, curve_fitting
    )
    return yield_curve_fit


def mbono_market_snapshot(
    df_input, valuation_date=ql.Date.todaysDate(), id_type="BBGID"
):
    """Obtain a snapshot of the market for a list of instruments. The
    instruments are identified by the id_type argument. The valuation date
    is used to calculate the settlement date of the bonds."""
    df_output = df_input.copy()
    instrument_ids = list(df_input.index)
    str_ids = "(" + str(instrument_ids)[1:-1] + ")"
    con = sqlite3.connect("../../db/MBONOdata.db")
    query = """SELECT %s, Name, PricingDate, Maturity
    FROM Instruments  WHERE %s in %s;""" % (
        id_type,
        id_type,
        str_ids,
    )
    df_output = df_output.join(
        pd.read_sql_query(
            query, con, parse_dates=["PricingDate", "Maturity"], index_col=id_type
        )
    )
    settlement_date = CALENDAR.advance(
        valuation_date, ql.Period(SETTLEMENT_DAYS, ql.Days)
    )
    bonds_sample = define_ql_bonds_bulk(instrument_ids, id_type=id_type)
    df_output["QL bond"] = bonds_sample
    for b in df_output.index:
        bond = df_output.loc[b, "QL bond"]
        df_output.loc[b, "Yield"] = bond.bondYield(
            df_output.loc[b, "Price"],
            DAY_COUNT,
            ql.Compounded,
            YIELD_BASIS,
            settlement_date,
        )
    return df_output


def fit_to_data(df_input, valuation_date=ql.Date.todaysDate()):
    """Fit a curve to a dataframe of bond prices. The dataframe should have
    columns 'Price' and 'QL bond' with the bond prices and QuantLib bond"""
    prices = list(df_input["Price"])
    bonds_sample = list(df_input["QL bond"])
    yield_curve_fit = fitted_ql_curve(
        bonds_sample, prices, valuation_date=valuation_date
    )
    yield_curve_fit.enableExtrapolation()
    return yield_curve_fit


def evaluate_bonds(df_input, yield_curve_fit, valuation_date=ql.Date.todaysDate()):
    """Evaluate a dataframe of bond prices with a given curve. The dataframe
    should have columns 'Price' and 'QL bond' with the bond prices and
    QuantLib bond"""
    df_output = df_input.copy()
    ql.Settings.instance().evaluationDate = valuation_date
    settlement_date = CALENDAR.advance(
        valuation_date, ql.Period(SETTLEMENT_DAYS, ql.Days)
    )
    disc_curve_handle = ql.YieldTermStructureHandle(yield_curve_fit)
    bond_engine = ql.DiscountingBondEngine(disc_curve_handle)
    for b in df_output.index:
        bond = df_output.loc[b, "QL bond"]
        bond.setPricingEngine(bond_engine)
        df_output.loc[b, "Th Price"] = bond.NPV() - bond.accruedAmount(settlement_date)
        df_output.loc[b, "Accrued"] = bond.accruedAmount(settlement_date)
        df_output.loc[b, "Th Yield"] = bond.bondYield(
            df_output.loc[b, "Th Price"], DAY_COUNT, ql.Compounded, YIELD_BASIS
        )
    df_output["Difference"] = df_output["Yield"] - df_output["Th Yield"]
    return df_output


def clean_fit(
    df_input, valuation_date=ql.Date.todaysDate(), min_days=270, outlier_range=0.001
):
    """Clean a dataframe of bond prices and fit a curve to the remaining
    data. The dataframe should have columns 'Price' and 'QL bond' with the
    bond prices and QuantLib bond"""
    df_input = df_input[
        (df_input["PricingDate"].apply(ql.Date().from_date) - valuation_date) < 0
    ].copy()
    df_input = df_input[
        (df_input["Maturity"].apply(ql.Date().from_date) - valuation_date) > 0
    ].copy()
    df_output = df_input[
        (df_input["Maturity"].apply(ql.Date().from_date) - valuation_date) > min_days
    ].copy()
    while True:
        yield_curve_fit = fit_to_data(df_output, valuation_date=valuation_date)
        df_output = evaluate_bonds(
            df_output, yield_curve_fit, valuation_date=valuation_date
        )
        if df_output["Difference"].abs().max() <= outlier_range:
            break
        else:
            df_output = df_output.drop(
                df_output[df_output["Difference"].abs() > outlier_range].index
            )
    try:
        df_output = evaluate_bonds(
            df_input, yield_curve_fit, valuation_date=valuation_date
        )
    except:
        print("Evaluation N/A")

    return yield_curve_fit, df_output


def zero_df_curve(
    yield_curve, tenors, day_count=ql.Actual365Fixed(), compounding=ql.Continuous
):
    """Create a dataframe with the zero rates of a yield curve for a set of
    tenors. The day_count and compounding arguments are used to calculate the
    zero rates."""
    zero_curve = pd.DataFrame(
        index=[str(i) + "Y" for i in tenors], columns=["Date", "Rate", "Years"]
    )
    ref_date = yield_curve.referenceDate()
    sample_dates = [ref_date + ql.Period(i, ql.Years) for i in tenors]
    sample_rates = [
        yield_curve.zeroRate(d, day_count, compounding).rate() for d in sample_dates
    ]
    zero_curve["Years"] = tenors
    zero_curve["Date"] = sample_dates
    zero_curve["Rate"] = sample_rates
    return zero_curve
