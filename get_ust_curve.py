import pandas as pd
import ustreasurycurve as ustcurve
import datetime as dt

dateT0 = '2024-05-01'
dateT0_dt = dt.datetime.strptime(dateT0, "%Y-%m-%d")

years = 1
d = dateT0_dt - dt.timedelta(days = years * 365)

d = d.strftime("%Y-%m-%d")

ustcurve = ustcurve.nominalRates(d, dateT0)

ustcurve.rename(columns = {'1m':'0.083y', '2m': '0.167y', '3m': '0.25y', '6m': '0.5y'}, inplace = True)

melted_ustcurve = ustcurve.melt(id_vars=['date'], var_name='risk_factor', value_name='value')

on_rate = melted_ustcurve[melted_ustcurve['risk_factor'] == '0.083y'].copy()
on_rate['risk_factor'] = round(1/365, 5)
on_rate['risk_factor'] = on_rate['risk_factor'].astype(str)
on_rate['risk_factor'] += 'y'
on_rate['value'] = pd.to_numeric(on_rate['value'], errors='coerce')
on_rate['value'] = on_rate['value'].round(3)

df_usdcurve = pd.concat([melted_ustcurve, on_rate])

df_usdcurve['period'] = 'USD_IR_' + df_usdcurve['risk_factor'].str.strip().str[:-1]

df_usdcurve['value'] *= 100

df_usdcurve = df_usdcurve[~(df_usdcurve['risk_factor'].isin(['0.167y', '10y', '20y', '30y']))]

df_usdcurve.drop(['risk_factor'], axis=1)
df_usdcurve.rename(columns = {'date':'tradedate'}, inplace = True)
df_usdcurve['ccy'] = 'USD'

file_path = f'/Users/mihailzaytsev/Desktop/CCR/market_data/usd_ir_{years}y.xlsx'
writer = pd.ExcelWriter(file_path, engine='xlsxwriter')
df_usdcurve.to_excel(writer, sheet_name='tcurve', index=False)
writer._save()

# Display the fetched data
print(ustcurve)
