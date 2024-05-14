import pandas as pd
import numpy as np
import math
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import statsmodels.api as sm

def prepare_discount_factors(St_d):
    St_d = St_d.sort_values(by=['risk_factor'])
    discount_factors = St_d[~(St_d['risk_factor'].isin(['USD', 'EUR', 'RUB']))].copy()
    discount_factors['tenor'] = discount_factors['risk_factor'].str.strip().str[7:]
    discount_factors['tenor'] = discount_factors['tenor'].astype(float)
    discount_factors['ccy'] = discount_factors['risk_factor'].str.strip().str[:3]
    discount_factors['gap_date'] = discount_factors.apply(lambda row: row['forward_date'] + datetime.timedelta(days=round(365 * row['tenor'])), axis=1)
    #discount_factors = discount_factors.sort_values(by=['gap_date'])
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
    #deals['days'] = (deals['mat_date'] - d).dt.days #проверить, и если не используется, снести
    deals['forward_date'] = d
    deals = join_df(deals, St_d, cf_date_type='start_date', suffix = '_st')
    deals = join_df(deals, St_d, cf_date_type='mat_date', suffix = '_end')
    deals['forward_rate'] = (deals['discount_factor_st']/deals['discount_factor_end'] - 1) * 365 / (deals['mat_date'] - (deals['start_date'])).dt.days

    #I'm terribly sorry for the following hardcode, but I was very tired and run out of time to difine a function fixing rates on fixed legs

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
    deals['days'] = (deals['mat_date'] - d).dt.days
    deals = join_df(deals, St_d, cf_date_type='mat_date')
    deals = join_fx(deals, St_d)
    deals['mtm'] = deals['amt'] * deals['discount_factor_fxd'] * deals['value']
    deals = deals.groupby(['client', 'csa'], as_index=False).agg({'mtm': 'sum'}).reset_index()   #отсюда убрал продукт
    return deals

def parce_rf(St, col_names, report_date):
    St_df = pd.DataFrame(St, columns=col_names)
    melted_St = St_df.melt(id_vars=['time_delta'], var_name='risk_factor', value_name='value')
    melted_St.reset_index(drop=True, inplace=True)
    melted_St['forward_date'] = melted_St.apply(lambda row: report_date + datetime.timedelta(days=round(365 * row['time_delta'])), axis=1)
    melted_St.drop('time_delta', axis=1, inplace=True)
    return melted_St

#get deals
deals = pd.read_excel('/Users/mihailzaytsev/Desktop/Диссер_мага/deals_for_GBM_IRS_3.xlsx', engine='openpyxl')
deals['mat_date'] = pd.to_datetime(deals['mat_date'])

#get market data
data_ir_rub = pd.read_excel('/Users/mihailzaytsev/Desktop/Диссер_мага/rub_ir_10y.xlsx', engine='openpyxl')
data_ir_rub['tradedate'] = pd.to_datetime(data_ir_rub['tradedate']).dt.date
data_ir_usd = pd.read_excel('/Users/mihailzaytsev/Desktop/Диссер_мага/usd_ir_10y.xlsx', engine='openpyxl')
data_ir_usd['tradedate'] = pd.to_datetime(data_ir_usd['tradedate']).dt.date
data_fx = pd.read_excel('/Users/mihailzaytsev/Desktop/Диссер_мага/fx_rates_10y.xlsx', engine='openpyxl')

data_ir = pd.concat([data_ir_rub, data_ir_usd])

dates_df = set(data_ir['tradedate'])
dates_df = pd.DataFrame(dates_df, columns = ['tradedate'])
dates_df = dates_df.sort_values('tradedate')

period_list = list(set(data_ir['period']))

for p in period_list: #to get risk factors in matrix form
    p_df = data_ir[data_ir['period'] == p].copy()
    p_df = p_df.rename(columns = {'value': p})
    p_df.drop(['tradetime', 'period', 'ccy', 'risk_factor'], axis= 1, inplace= True)
    dates_df = pd.merge(dates_df, p_df, how = 'inner', on = 'tradedate')

ir_df = dates_df.copy()

usd_rub = (data_fx[data_fx['secid'] == 'USDFIXME']).rename(columns = {'rate': 'USD'})
usd_rub.drop('secid', axis= 1 , inplace= True )
eur_rub = (data_fx[data_fx['secid'] == 'EURFIXME']).rename(columns = {'rate': 'EUR'})
eur_rub.drop('secid', axis= 1 , inplace= True )

fx_df = pd.merge(usd_rub, eur_rub, how = 'inner', on = 'tradedate')
fx_df['tradedate'] = pd.to_datetime(fx_df['tradedate']).dt.date
num_fx_rf = len(fx_df.columns) - 1
all_df = pd.merge(fx_df, dates_df, how = 'inner', on = 'tradedate')
report_date = all_df['tradedate'].iloc[-1]
#report_date = datetime.datetime.strptime(report_date, '%Y-%m-%d')

all_df.drop('tradedate', axis= 1, inplace= True)
fx_df.drop('tradedate', axis= 1, inplace= True)
all_df_col_names = all_df.columns.tolist()
all_df_col_names.append('time_delta')

fx_log_returns = all_df.iloc[:, :num_fx_rf].apply(lambda x: (np.log(x / x.shift(1))).fillna(0), axis=0)
ir_simple_returns = all_df.iloc[:, num_fx_rf:].apply(lambda x: (x / x.shift(1) - 1).fillna(0), axis=0)

df_returns = pd.concat([fx_log_returns, ir_simple_returns], axis=1)

def volatility_rescaling(returns, wind):
    local_volatilities = returns.rolling(window=wind).std()
    local_volatilities = local_volatilities.drop(local_volatilities.index[:wind])
    returns = returns.drop(local_volatilities.index[:wind])
    T = (pd.DataFrame(local_volatilities.iloc[-1])).T
    T_parsed = pd.concat([T]*len(returns.index), ignore_index=True)
    local_volatilities = local_volatilities.reset_index(drop = True)
    returns = returns.reset_index(drop=True)
    T_parsed = T_parsed.reset_index(drop=True)
    rescaled_returns = returns/local_volatilities*T_parsed
    rescaled_returns = rescaled_returns.fillna(0.001)
    rescaled_returns.iloc[0] = 0
    return rescaled_returns

# plt.plot(df_returns['RUB_IR_0.25'])
# plt.show()

return_rescaling = True

if return_rescaling == True:
    df_returns = volatility_rescaling(df_returns, wind = 20)

# plt.plot(df_returns['RUB_IR_0.25'])
# plt.show()

#####################
# define necessary parametres of input market data

num_sims = 3
years = 2
steps = 52 * years + 1
steps = int(steps)
num_of_risk_factors = df_returns.shape[1]

fx_log_returns.replace([np.inf, -np.inf], np.nan, inplace=True)
fx_log_returns.dropna(inplace=True)

fx_meanReturns = fx_log_returns.mean() * 5
fx_sdrt_div = fx_log_returns.std() * math.sqrt(5)
fx_0 = ((fx_df.iloc[-1]).to_numpy()).T

covMatrix = df_returns.cov()
corrMatrix = df_returns.corr()

# here changed df_returns =>
dt = 1/((steps-1)/years)

#Vasicek parametres determination
ir = all_df.iloc[:, num_fx_rf:].copy()
delta = (ir - ir.shift(1)).fillna(0)
lag = ir.shift(1).fillna(0)

parametres_df = pd.DataFrame()
for c in ir.columns.tolist():
    x = lag[c]
    y = delta[c]
    x = sm.add_constant(x)
    model = sm.OLS(y, x)
    results = model.fit()
    slope = results.params[1]  # params[1] - это slope
    intercept = results.params[0]# params[0] - это intercept
    sey = np.sqrt(results.scale)
    p_value = results.pvalues[1]
    parametres = dict({'risk_factor': c, 'a': -slope, 'b': intercept/(-slope), 'sigma': sey, 'p_value': p_value})
    if p_value > 0.05:
        print('p value of ', c, ' is ', p_value, '(less then 0.05)')
    parametres = pd.DataFrame([parametres])
    parametres_df = pd.concat([parametres_df, parametres])

parametres_df.set_index("risk_factor", inplace = True)

ir_a = parametres_df['a']
ir_b = parametres_df['b']
ir_sigma = parametres_df['sigma']
ir_0 = ((ir.iloc[-1] * np.exp(-ir_a * dt)).to_numpy()) #.T наверно понадобится транспонирование
ir_drift = (ir_b * (1-np.exp(-ir_a * dt)))
ir_0 = ir_0.reshape(1, -1)
fx_0 = fx_0.reshape(1, -1)
S0 = np.concatenate((fx_0, ir_0), axis=1)

def adjust_matrix_to_positive_definite(matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    min_positive = eigenvalues[eigenvalues > 0].min()
    epsilon = min_positive * 10
    eigenvalues[eigenvalues <= 0] = epsilon
    eigenvalues[eigenvalues <= 0.0001] = epsilon
    adjusted_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    eigenvalues, eigenvectors = np.linalg.eigh(adjusted_matrix)
    return adjusted_matrix

success = 0
while success == 0:
    try:
        chol = np.linalg.cholesky(corrMatrix)  # get cholezky decomposition to mind the correlation of fx and ir epsilons
        success = 1
    except:
        corrMatrix = adjust_matrix_to_positive_definite(corrMatrix)
        print('correlation Matrix was adjusted for Cholezky decomposition')

#convert to arrays
fx_mean_returns_col = fx_meanReturns.to_numpy().reshape(1, -1)
fx_sdrt_div_col = fx_sdrt_div.to_numpy().reshape(1, -1)
fx_drift = (fx_mean_returns_col - fx_sdrt_div_col ** 2 / 2) * dt
fx_drift = fx_drift.reshape(1, -1)
#fx_drift[fx_drift < 1e-4] = 0.001
ir_drift_col = ir_drift.to_numpy().reshape(1, -1)
drift = np.concatenate((fx_drift, ir_drift_col), axis=1)
drift_reshaped = (np.tile(drift, (steps - 1, 1))).T
ir_sdrt_div = np.sqrt(ir_sigma**2 * (1 - np.exp(-2 * ir_a * dt)) / (2 * ir_a))
fx_sdrt_div_col = fx_sdrt_div_col.reshape(1, -1)
ir_sdrt_div = ir_sdrt_div.to_numpy().reshape(1, -1)
#ir_sdrt_div = ir_sdrt_div.T
sdrt_div = np.concatenate((fx_sdrt_div_col, ir_sdrt_div), axis=1)

n = 0
st = time.time()
St_full = np.empty([steps, num_of_risk_factors])
mtm_generated = pd.DataFrame()
while n < num_sims:
    #generate standard normal non-correlated outcomes
    uncrlted_eps = np.random.normal(size=(steps-1, df_returns.shape[1]))

    #correlate them through cholezky matrix
    crlted_eps = (chol @ uncrlted_eps.T) #абвгд

    #some adjustments
    num_columns = steps - 1 #crlted_eps.shape[1]
    sdrt_div_col_reshaped = (np.tile(sdrt_div, (num_columns, 1))).T #sdrt_div_col

    #calculation of gbm
    stochastic = crlted_eps * sdrt_div_col_reshaped * np.sqrt(dt)
    ito_process = drift_reshaped + stochastic
    St = (np.exp(ito_process)).T
    St = np.vstack([np.ones(num_of_risk_factors), St])
    St = S0 * St.cumprod(axis=0)

    # Define t interval correctly
    t = np.linspace(0, years, steps)
    t_column = t.reshape(-1, 1)

    # Require numpy array that is the same shape as St
    tt = np.full(shape=(num_of_risk_factors, steps), fill_value=t).T

# #риск факторы на разных осях. !!! Номера хардкодом !!!
#     plt.figure()
#
#     # Создаем первый график для первых двух столбцов St
#     ax1 = plt.gca()  # Получаем текущие оси
#     ax1.plot(tt, St[:, 0], label='USD_FX_rate')  # Первый столбец данных
#     ax1.plot(tt, St[:, 1], label='EUR_FX_rate')  # Второй столбец данных
#     ax1.set_xlabel("Время в годах $(t)$")
#     ax1.set_ylabel("Валютные риск-факторы $(S_t)$")
#
#     # Создаем вторичные оси для оставшихся 25 столбцов St
#     ax2 = ax1.twinx()  # Создаем вторичную ось Y
#     for i in range(2, 27):
#         ax2.plot(tt, St[:, i], label=f'IR_ {i + 1}', alpha=0.5)  # Остальные переменные с прозрачностью
#
#     ax2.set_ylabel("Процентные риск-факторы $(S_t)$")
#     # Для большей наглядности можно добавить легенду на вторичной оси, если нужно
#
#     # Добавляем название графика
#     plt.title(
#         "Реализация геометрического броуновского движения\n с разложением Холецкого"
#     )
#
#     plt.show()


# #общий график для всех риск-факторов
#
#     plt.plot(tt, St)
#     plt.xlabel("Years $(t)$")
#     plt.ylabel("Risk Factirs $(S_t)$")
#     plt.title(
#        "Realizations of Geometric Brownian Motion\n $dS_t = \mu S_t dt + \sigma S_t dW_t$\n"
#     )
#     plt.show()

    St = np.concatenate((St, t_column), axis = 1)

    St = parce_rf(St, all_df_col_names, report_date)
    St = main(St, deals)
    n += 1
    print(n)
    St['nums'] = n
    mtm_generated = pd.concat([mtm_generated, St])

et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')

#pivot_df = mtm_generated.pivot_table(index='nums', columns='forward_date', values='mtm', aggfunc='sum')
#mtm_generated['forward_date'] = mtm_generated['forward_date'].dt.strftime('%d.%m.%Y')

mtm_generatied_total_bank = mtm_generated.groupby(['forward_date', 'nums'], as_index=False).agg({'mtm': 'sum'}).reset_index()

melted_df = mtm_generatied_total_bank.pivot(index='nums', columns='forward_date', values='mtm').reset_index().melt(id_vars='nums', var_name='forward_date', value_name='mtm')
mtm_generated = mtm_generated.sort_values(by=['forward_date', 'nums'])


# Sort the DataFrame by 'forward_date' for proper plotting of line segments
melted_df.sort_values(by=['nums', 'forward_date'], inplace=True)

# Plotting line segments
for num in melted_df['nums'].unique():
    data_subset = melted_df[melted_df['nums'] == num]
    plt.plot(data_subset['forward_date'], data_subset['mtm'], marker='o', label=f'nums={num}')

plt.xlabel('forward_date')
plt.ylabel('mtm')
plt.title('PFE of Derivatives portfolio')
#plt.legend()
plt.grid(True)
plt.show()

#determine time tenors on which you are going to price MtM + PFE

forward_dates_list = list(set(mtm_generated['forward_date']))
forward_dates_list.sort()

valid_values = []

index_list = [1, 4, 13, 26, 52, 261, 521]

for index in index_list:
    if len(forward_dates_list) > index:
        valid_values.append(forward_dates_list[index])

mtm_generated_with_tenors = mtm_generated[mtm_generated['forward_date'].isin(valid_values)].copy()

grand_final = pd.DataFrame()
client_list = list(set(mtm_generated_with_tenors['client']))
for client in client_list:
    exp_of_the_client = mtm_generated_with_tenors[mtm_generated_with_tenors['client'] == client].copy()
    exp_of_the_client = exp_of_the_client.sort_values(by='forward_date')
    exp_of_the_client['time_bucket'] = exp_of_the_client['forward_date'].rank(method='dense').astype(int)
    exp_of_the_client.loc[exp_of_the_client['time_bucket'] == 1, 'time_bucket'] = 0.01923
    exp_of_the_client.loc[exp_of_the_client['time_bucket'] == 2, 'time_bucket'] = 0.08333
    exp_of_the_client.loc[exp_of_the_client['time_bucket'] == 3, 'time_bucket'] = 0.25
    exp_of_the_client.loc[exp_of_the_client['time_bucket'] == 4, 'time_bucket'] = 0.5
    exp_of_the_client.loc[exp_of_the_client['time_bucket'] == 5, 'time_bucket'] = 1
    exp_of_the_client.loc[exp_of_the_client['time_bucket'] == 6, 'time_bucket'] = 5
    exp_of_the_client.loc[exp_of_the_client['time_bucket'] == 7, 'time_bucket'] = 10
    dates_list = list(set(exp_of_the_client['forward_date']))
    client_metrics = pd.DataFrame()
    for date in dates_list:
        exp_of_the_client_on_date = exp_of_the_client[exp_of_the_client['forward_date'] == date].copy()
        positive_exp = exp_of_the_client_on_date[exp_of_the_client_on_date['mtm'] >= 0]
        negative_exp = exp_of_the_client_on_date[exp_of_the_client_on_date['mtm'] <= 0]
        pfe = positive_exp['mtm'].quantile(0.99)
        epe = positive_exp['mtm'].quantile(0.5)
        ene = negative_exp['mtm'].quantile(0.5)
        time_bucket = exp_of_the_client_on_date['time_bucket'].iloc[0]
        result = dict({'client': client, 'date': date, 'time_bucket': time_bucket, 'pfe': pfe, 'epe': epe, 'ene': ene})
        result = pd.DataFrame([result])
        client_metrics = pd.concat([client_metrics, result], ignore_index=True)
    client_metrics_gr = client_metrics.groupby(['client'], as_index=False).agg({'pfe': 'max', 'epe': 'max', 'ene': 'min'}).reset_index()
    grand_final = pd.concat([grand_final, client_metrics_gr], ignore_index=True)

client_metrics.to_csv('client_metrics.csv')

print(grand_final)


