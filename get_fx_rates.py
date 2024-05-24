import pandas as pd
import json
import requests
from urllib import parse
import datetime as dt
import yfinance as yf
import xlsxwriter
import csv

def get_fx(start_date, end_date, years, ticker):
    start_date = start_date.strftime("%Y-%m-%d")
    start = 0
    fx_df = pd.DataFrame()
    while start < years*365:
        url = f'https://iss.moex.com/iss/statistics/engines/currency/markets/fixing/{ticker}.json?from={start_date}&till={end_date}&start={start}'
        response = requests.get(url)
        result = json.loads(response.text)
        table = 'history'
        col_name = result[table]['columns']
        resp_date = result[table]['data']
        fx = pd.DataFrame(resp_date, columns=col_name)
        start += 100
        fx_df = pd.concat([fx_df, fx])
    return fx_df

dateT0_str = '2024-05-01'
dateT0_dt = dt.datetime.strptime(dateT0_str, "%Y-%m-%d")

years = 1
d = dateT0_dt - dt.timedelta(days = years * 365)

EURRUB = get_fx(d, dateT0_str, years, 'EURFIXME')
USDRUB = get_fx(d, dateT0_str, years, 'USDFIXME')

ccy_df = pd.concat([EURRUB, USDRUB])
ccy_df = ccy_df.sort_values(by=['tradedate'])

file_path = f'/Users/mihailzaytsev/Desktop/CCR/market_data/fx_rates_{years}y.xlsx'
writer = pd.ExcelWriter(file_path, engine='xlsxwriter')
ccy_df.to_excel(writer, sheet_name='fx', index=False)
writer._save()

print("Data exported")
