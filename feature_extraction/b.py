import pandas as pd
import gc
from utils import *


class Bureau:

  def __init__(self, debug=False, fill_missing=False):
    self.b = reduce_mem_usage(read_data('bureau', debug=debug))
    self.fill_missing = fill_missing
    self.df = pd.DataFrame({'SK_ID_CURR': self.b['SK_ID_CURR'].unique()})


  def execute(self):
    self.clean()
    self.preprocess()
    self.handcrafted()
    self.aggregations()  # 22 features
    for col in list(self.df):
      if col != 'SK_ID_CURR':
        self.df.rename(columns={col: 'B_'+ col}, inplace=True)
    del self.b
    gc.collect()
    print('Bureau Shape', self.df.shape)
    return self.df

  def clean(self):
    self.b['DAYS_CREDIT_ENDDATE'][self.b['DAYS_CREDIT_ENDDATE'] < -40000] = np.nan
    self.b['DAYS_CREDIT_UPDATE'][self.b['DAYS_CREDIT_UPDATE'] < -40000] = np.nan
    self.b['DAYS_ENDDATE_FACT'][self.b['DAYS_ENDDATE_FACT'] < -40000] = np.nan
    if self.fill_missing:
        self.b['AMT_CREDIT_SUM'].fillna(self.fill_value, inplace=True)
        self.b['AMT_CREDIT_SUM_DEBT'].fillna(self.fill_value, inplace=True)
        self.b['AMT_CREDIT_SUM_OVERDUE'].fillna(self.fill_value, inplace=True)
        self.b['CNT_CREDIT_PROLONG'].fillna(self.fill_value, inplace=True)


  def preprocess(self):
    self.b['bureau_credit_active_binary'] = (self.b['CREDIT_ACTIVE'] != 'Closed').astype(int)
    self.b['bureau_credit_enddate_binary'] = (self.b['DAYS_CREDIT_ENDDATE'] > 0).astype(int)

  def aggregations(self):
    num_aggregations = {
        'AMT_CREDIT_SUM':['mean', 'sum'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'], ## THIS NEEDS INVESTIGATION -
        'SK_ID_BUREAU': ['count'],
        'DAYS_CREDIT_ENDDATE':['mean', 'sum'],
        'DAYS_CREDIT': ['mean', 'min', 'var', 'max'],
        'AMT_CREDIT_SUM_DEBT': ['mean', 'sum'],
        'AMT_CREDIT_SUM_LIMIT': ['mean'],
        #'BB_LATEST_MONTHS_BALANCE': ['max'],
        #'BB_LATEST_STATUS': ['first'],
        'DAYS_CREDIT_UPDATE':['mean'],
        'CREDIT_DAY_OVERDUE': ['nunique'],
        'CNT_CREDIT_PROLONG': ['sum', 'max'],
        'AMT_CREDIT_SUM_OVERDUE':['sum', 'mean'],
    }
    b_g = self.b.groupby('SK_ID_CURR').agg({**num_aggregations})
    b_g.columns = pd.Index(['B_' + e[0] + "_" + e[1].upper() for e in b_g.columns.tolist()])
    b_g['SK_ID_CURR'] = b_g.index
    self.df = self.df.merge(b_g, on=['SK_ID_CURR'], how='left')
    del b_g
    gc.collect()

  def handcrafted(self):
    groupby = self.b.groupby(by=['SK_ID_CURR'])

    g = groupby['DAYS_CREDIT'].agg('count').reset_index()
    g.rename(index=str, columns={'DAYS_CREDIT': 'DAYS_CREDIT_count'}, inplace=True)
    self.df = self.df.merge(g, on=['SK_ID_CURR'], how='left')

    g = groupby['CREDIT_TYPE'].agg('nunique').reset_index()
    g.rename(index=str, columns={'CREDIT_TYPE': 'CREDIT_TYPE_nunique'}, inplace=True)
    self.df = self.df.merge(g, on=['SK_ID_CURR'], how='left')

    g = groupby['bureau_credit_active_binary'].agg('mean').reset_index()
    g.rename(index=str, columns={'bureau_credit_active_binary': 'credit_active_binary'}, inplace=True)
    self.df = self.df.merge(g, on=['SK_ID_CURR'], how='left')

    g = groupby['AMT_CREDIT_SUM_DEBT'].agg('sum').reset_index()
    g.rename(index=str, columns={'AMT_CREDIT_SUM_DEBT': 'total_customer_debt'}, inplace=True)
    self.df = self.df.merge(g, on=['SK_ID_CURR'], how='left')

    g = groupby['AMT_CREDIT_SUM'].agg('sum').reset_index()
    g.rename(index=str, columns={'AMT_CREDIT_SUM': 'total_customer_credit'}, inplace=True)
    self.df = self.df.merge(g, on=['SK_ID_CURR'], how='left')

    g = groupby['AMT_CREDIT_SUM_OVERDUE'].agg('sum').reset_index()
    g.rename(index=str, columns={'AMT_CREDIT_SUM_OVERDUE': 'total_customer_overdue'}, inplace=True)
    self.df = self.df.merge(g, on=['SK_ID_CURR'], how='left')

    g = groupby['CNT_CREDIT_PROLONG'].agg('sum').reset_index()
    g.rename(index=str, columns={'CNT_CREDIT_PROLONG': 'average_creditdays_prolonged'}, inplace=True)
    self.df = self.df.merge(g, on=['SK_ID_CURR'], how='left')

    g = groupby['bureau_credit_enddate_binary'].agg('mean').reset_index()
    g.rename(index=str, columns={'bureau_credit_enddate_binary': 'credit_enddate_percentage'}, inplace=True)
    self.df = self.df.merge(g, on=['SK_ID_CURR'], how='left')

    self.df['average_of_past_loans_per_type'] =  self.df['DAYS_CREDIT_count'] / self.df['CREDIT_TYPE_nunique']
    self.df['debt_credit_ratio'] = self.df['total_customer_debt'] / self.df['total_customer_credit']
    self.df['overdue_debt_ratio'] = self.df['total_customer_overdue'] / self.df['total_customer_debt']

    del g
    gc.collect()
