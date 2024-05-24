import pandas as pd
import json
import requests
from urllib import parse
import datetime as dt
import yfinance as yf
import xlsxwriter
import csv

def get_zcurve(date):
    date = date.strftime("%Y-%m-%d")
    url = f'http://iss.moex.com/iss/engines/stock/zcyc.json?date={date}'
    response = requests.get(url)
    result = json.loads(response.text)
    table = 'yearyields'
    col_name = result[table]['columns']
    resp_date = result[table]['data']
    zero_coupon_yield_curve = pd.DataFrame(resp_date, columns=col_name)
    return zero_coupon_yield_curve

dateT0_str = '2024-05-01'
dateT0_dt = dt.datetime.strptime(dateT0_str, "%Y-%m-%d")

years = 1
d = dateT0_dt - dt.timedelta(days = years * 365)

gcurve_data = pd.DataFrame()

while d < dateT0_dt + dt.timedelta(days = 1):
    print(d)
    gcurve = get_zcurve(d)
    gcurve_data = pd.concat([gcurve_data, gcurve])
    d += dt.timedelta(days = 1)

on_gcurve_data = gcurve_data[gcurve_data['period'] == 0.25].copy()
on_gcurve_data['period'] = round(1/365, 5)
gcurve_data = pd.concat([gcurve_data, on_gcurve_data])

gcurve_data['ccy'] = 'RUB'
gcurve_data['period'] = 'RUB_IR_' + gcurve_data['period'].astype(str)

gcurve_data = gcurve_data[~(gcurve_data['period'].isin(['RUB_IR_10.0', 'RUB_IR_15.0', 'RUB_IR_20.0', 'RUB_IR_30.0']))]

file_path = f'/Users/mihailzaytsev/Desktop/CCR/market_data/rub_ir_{years}y.xlsx'
writer = pd.ExcelWriter(file_path, engine='xlsxwriter')
gcurve_data.to_excel(writer, sheet_name='gcurve', index=False)
writer._save()

print("Data exported")
