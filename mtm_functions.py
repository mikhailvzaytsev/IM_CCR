import pandas as pd
import numpy as np
import math
import datetime

def prepare_discount_factors(St_d):
    St_d = St_d.sort_values(by=['risk_factor'])
    discount_factors = St_d[~(St_d['risk_factor'].isin(['USD', 'EUR', 'RUB']))].copy()
    discount_factors['tenor'] = discount_factors['risk_factor'].str.strip().str[7:]
    discount_factors['tenor'] = discount_factors['tenor'].astype(float)
    discount_factors['ccy'] = discount_factors['risk_factor'].str.strip().str[:3]
    discount_factors['gap_date'] = discount_factors.apply(lambda row: row['forward_date'] + datetime.timedelta(days=round(365 * row['tenor'])), axis=1)
    discount_factors['gap_date'] = pd.to_datetime(discount_factors['gap_date'])
    discount_factors['forward_date'] = pd.to_datetime(discount_factors['forward_date'])
    discount_factors['bidval'] = np.exp(-1 * discount_factors['value']/100*((discount_factors['gap_date'] - discount_factors['forward_date']).dt.days)/365)
    discount_factors['next_gap_date'] = discount_factors['gap_date'].shift(-1)
    discount_factors['next_risk_factor'] = discount_factors['risk_factor'].shift(-1)
    discount_factors['next_bidval'] = discount_factors['bidval'].shift(-1)
    discount_factors['next_bidval'] = discount_factors['next_bidval'].fillna(0)
    discount_factors = discount_factors.rename(columns={"gap_date": "prev_gap_date"})
    return discount_factors

def join_fx(deals, St):
    St = St[St['risk_factor'].isin(['USD', 'EUR'])].copy()
    deals['forward_date'] = pd.to_datetime(deals['forward_date'])
    St['forward_date'] = pd.to_datetime(St['forward_date'])
    deals = deals.merge(St[['forward_date', 'risk_factor', 'value']], how='left', left_on=['forward_date', 'ccy'], right_on=['forward_date', 'risk_factor']).drop('risk_factor', axis = 1)
    deals['value'] = deals['value'].fillna(1)
    return deals

def join_df(cash_flow_full, discount_factors, cf_date_type = 'mat_date', suffix = '_fxd'):
    discount_factors = prepare_discount_factors(discount_factors)
    discount_factors['forward_date'] = pd.to_datetime(discount_factors['forward_date'])
    cash_flow_full['forward_date'] = pd.to_datetime(cash_flow_full['forward_date'])
    cash_flow_full_df = cash_flow_full.merge(discount_factors[['bidval', 'next_bidval', 'ccy', 'forward_date', 'prev_gap_date', 'next_gap_date']],
                                                   how='left', left_on=['forward_date', 'ccy'],
                                          right_on = ['forward_date', 'ccy'])

    cash_flow_full_df = cash_flow_full_df[(cash_flow_full_df['prev_gap_date'] <= cash_flow_full_df['mat_date'])
                                                & (cash_flow_full_df['mat_date'] <
                                                   cash_flow_full_df['next_gap_date'])].rename(
                                                columns={"bidval": "bidval" + suffix,
                                                         "next_bidval": "next_bidval" + suffix,
                                                         "prev_gap_date": "prev_gap_date" + suffix,
                                                         "next_gap_date": "next_gap_date" + suffix})
    interpolate_df(cash_flow_full_df, cf_date_type, suffix)

    cash_flow_full_df.drop(["bidval" + suffix,
                            "next_bidval" + suffix,
                            "prev_gap_date" + suffix,
                            "next_gap_date" + suffix], axis= 1, inplace= True)

    return cash_flow_full_df

def interpolate_df(data, cf_date, suffix = ''):
    data['year_frac_t1'] = (data['prev_gap_date' + suffix] - (data['forward_date'])).dt.days / 365.0
    data['year_frac_t2'] = (data['next_gap_date' + suffix] - data['forward_date']).dt.days / 365.0
    data['year_frac_t0'] = (data[cf_date] - data['forward_date']).dt.days / 365.0
    data['z1'] = -(1 / data['year_frac_t1']) * np.log(data['bidval'  + suffix].astype(float))
    data['z2'] = -(1 / data['year_frac_t2']) * np.log(data['next_bidval'  + suffix].astype(float))
    data.loc[data['z1'].isnull(), 'z1'] = data.loc[data['z1'].isnull(), 'z2']
    data.loc[data['bidval' + suffix] == 1, 'z1'] = data.loc[data['bidval' + suffix] == 1, 'z2']
    data['z0'  + suffix] = (data['z2'] * (data['year_frac_t0'] - data['year_frac_t1']) +
                 data['z1']  * (data['year_frac_t2'] - data['year_frac_t0'])) / \
                           (data['year_frac_t2'] - data['year_frac_t1'])
    data['discount_factor' + suffix] = np.exp(-data['year_frac_t0'] * data['z0' + suffix])
    data['discount_factor' + suffix] = data['discount_factor' + suffix].fillna (1)

def main(St, deals):
    deals_all = pd.DataFrame()
    forward_date_list = list(set(St['forward_date']))
    forward_date_list.sort()
    for d in forward_date_list:
        #print(d)
        St_d = St[St['forward_date'] == d].copy()
        deals['forward_date'] = d
        ir_deals = deals[deals['product'].isin(['IRS', 'OIS', 'XCCY'])].copy() #add new instruments if needed
        fx_deals = deals[deals['product'].isin(['FXF', 'FXS', 'FXO'])].copy() #add new instruments if needed
        ir_deals_mtm = ir_mtm_calc(ir_deals, St_d, d)
        fx_deals_mtm = fx_mtm_calc(fx_deals, St_d, d)
        deals_all = pd.concat([deals_all, fx_deals_mtm, ir_deals_mtm])
    deals_csa = deals_all[deals_all['csa'] == 'Y'].copy()
    deals_all = deals_all[deals_all['csa'] == 'N']
    deals_csa_pfe_limited = csa_pfe_limitation(deals_csa, forward_date_list)
    deals_all = pd.concat([deals_all, deals_csa_pfe_limited])
    return deals_all

def csa_pfe_limitation(deals, forward_date_list):
    deals = deals.sort_values(by=['client', 'forward_date'])
    deals['mtm_adjusted'] = deals.groupby('client')['mtm'].transform(lambda x: x - x.iloc[0]) # because varmargin covers current MtM revaluation
    deals['mtm'] = deals['mtm_adjusted']
    deals.drop(['mtm_adjusted'], axis=1)
    today = forward_date_list[0]
    mpor_date = today + datetime.timedelta(days=14) #RISDA margin period of risk is 14 days
    mpor_date = pd.Timestamp(mpor_date)
    mask = deals['forward_date'] > mpor_date
    mpor_mtm = deals[deals['forward_date'] == mpor_date].set_index('client')['mtm']
    deals.loc[mask, 'mtm'] = deals.loc[mask, 'client'].map(mpor_mtm)
    return deals


def ir_mtm_calc(deals, St_d, d):
    deals['forward_date'] = d
    deals = join_df(deals, St_d, cf_date_type='start_date', suffix = '_st')
    deals = join_df(deals, St_d, cf_date_type='mat_date', suffix = '_end')
    deals['forward_rate'] = (deals['discount_factor_st']/deals['discount_factor_end'] - 1) * 365 / (deals['mat_date'] - (deals['start_date'])).dt.days

    #здесь конечно лучше бы сделать честный расчет, но это можно взять уже на будущее

    St_d_short_rates = St_d[St_d['risk_factor'].str.strip().str[-4:] == '0.25'].copy()

    short_term_rub_rate = St_d_short_rates[St_d_short_rates['risk_factor'].str.strip().str[:3] == 'RUB']
    short_term_rub_rate = short_term_rub_rate['value'].iloc[0].astype(float)/100
    deals.loc[
        (deals['fixed_deal'] == 'N') & (deals['ccy'] == 'RUB') & (deals['start_date'] <= deals['forward_date']), 'rate'] = short_term_rub_rate

    short_term_usd_rate = St_d_short_rates[St_d_short_rates['risk_factor'].str.strip().str[:3] == 'USD']
    short_term_usd_rate = short_term_usd_rate['value'].iloc[0].astype(float)/100
    deals.loc[
        (deals['fixed_deal'] == 'N') & (deals['ccy'] == 'USD') & (deals['start_date'] <= deals['forward_date']), 'rate'] = short_term_usd_rate

    ###

    deals.loc[(deals['start_date'] <= deals['forward_date']) | (deals['fixed_deal'] == 'Y'),
                    'cash_flow'] = deals['amt'] * deals['rate'] * (deals['mat_date'] - (deals['start_date'])).dt.days/365
    deals.loc[(deals['fixed_deal'] == 'N') &
                (deals['start_date'] > deals['forward_date']),
                    'cash_flow'] = deals['amt'] * deals['forward_rate'] * (deals['mat_date'] - (deals['start_date'])).dt.days/365
    deals = join_fx(deals, St_d)
    deals['mtm'] = deals['cash_flow'] * deals['discount_factor_end'] * deals['value']
    deals_grouped = deals.groupby(['client', 'csa', 'forward_date'], as_index=False).agg({'mtm': 'sum'}).reset_index()  # отсюда убрал продукт
    return deals_grouped


def fx_mtm_calc(deals, St_d, d):
    d = pd.Timestamp(d)
    deals['forward_date'] = d
    deals['days'] = (deals['mat_date'] - d).dt.days
    deals = join_df(deals, St_d, cf_date_type='mat_date')
    deals = join_fx(deals, St_d)
    deals['mtm'] = deals['amt'] * deals['discount_factor_fxd'] * deals['value']
    deals = deals.groupby(['client', 'csa', 'forward_date'], as_index=False).agg({'mtm': 'sum'}).reset_index()   #отсюда убрал продукт
    return deals
