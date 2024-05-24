import pandas as pd
import numpy as np
import math
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time


PD_matrix = pd.read_excel('/Users/mihailzaytsev/Desktop/Диссер_мага/PD.xlsx', engine='openpyxl')
PD_matrix.set_index('Rating', inplace=True)
LGD = pd.read_excel('/Users/mihailzaytsev/Desktop/Диссер_мага/LGD.xlsx', engine='openpyxl')
LGD.set_index('rating', inplace=True)
client_metrics = pd.read_csv('/Users/mihailzaytsev/Desktop/practical-python/Work/Data/client_metrics.csv', encoding='latin1')
client_rating = pd.read_excel('/Users/mihailzaytsev/Desktop/Диссер_мага/Client_ratings.xlsx', engine='openpyxl')

PD_matrix = PD_matrix.T

PD = PD_matrix.diff()
PD.iloc[0] = PD_matrix.iloc[0]
PD = PD.reset_index().rename(columns={'index': 'time_bucket'})
#melted_St = St_df.melt(id_vars=['time_delta'], var_name='risk_factor', value_name='value')

PD_melted = PD.melt(id_vars=['time_bucket'], var_name='rating', value_name='dPD')

client_metrics = client_metrics.merge(client_rating, how='left', left_on = 'client', right_on = 'client_id')
client_metrics = client_metrics.merge(PD_melted, how='left', on = ['rating', 'time_bucket'])
client_metrics = client_metrics.merge(LGD, how='left', on = 'rating')

cva = client_metrics.groupby(['client'], as_index=False).apply(lambda x:
                                                               pd.Series({
                                                                   'cva': (x['epe'] * x['dPD'] * x['LGD']).sum()
                                                               })).reset_index()

print(cva)