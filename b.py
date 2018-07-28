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
    self.aggregations()
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

  def handcrafted(self):
    pass

  def aggregations(self):
        groupby = self.b.groupby(by=['SK_ID_CURR'])

        g = groupby['DAYS_CREDIT'].agg('count').reset_index()
        g.rename(index=str, columns={'DAYS_CREDIT': 'bureau_number_of_past_loans'}, inplace=True)
        self.df = self.df.merge(g, on=['SK_ID_CURR'], how='left')

        g = groupby['CREDIT_TYPE'].agg('nunique').reset_index()
        g.rename(index=str, columns={'CREDIT_TYPE': 'bureau_number_of_loan_types'}, inplace=True)
        self.df = self.df.merge(g, on=['SK_ID_CURR'], how='left')

        g = groupby['bureau_credit_active_binary'].agg('mean').reset_index()
        g.rename(index=str, columns={'bureau_credit_active_binary': 'bureau_credit_active_binary'}, inplace=True)
        self.df = self.df.merge(g, on=['SK_ID_CURR'], how='left')

        g = groupby['AMT_CREDIT_SUM_DEBT'].agg('sum').reset_index()
        g.rename(index=str, columns={'AMT_CREDIT_SUM_DEBT': 'bureau_total_customer_debt'}, inplace=True)
        self.df = self.df.merge(g, on=['SK_ID_CURR'], how='left')

        g = groupby['AMT_CREDIT_SUM'].agg('sum').reset_index()
        g.rename(index=str, columns={'AMT_CREDIT_SUM': 'bureau_total_customer_credit'}, inplace=True)
        self.df = self.df.merge(g, on=['SK_ID_CURR'], how='left')

        g = groupby['AMT_CREDIT_SUM_OVERDUE'].agg('sum').reset_index()
        g.rename(index=str, columns={'AMT_CREDIT_SUM_OVERDUE': 'bureau_total_customer_overdue'}, inplace=True)
        self.df = self.df.merge(g, on=['SK_ID_CURR'], how='left')

        g = groupby['CNT_CREDIT_PROLONG'].agg('sum').reset_index()
        g.rename(index=str, columns={'CNT_CREDIT_PROLONG': 'bureau_average_creditdays_prolonged'}, inplace=True)
        self.df = self.df.merge(g, on=['SK_ID_CURR'], how='left')

        g = groupby['bureau_credit_enddate_binary'].agg('mean').reset_index()
        g.rename(index=str, columns={'bureau_credit_enddate_binary': 'bureau_credit_enddate_percentage'}, inplace=True)
        self.df = self.df.merge(g, on=['SK_ID_CURR'], how='left')

        self.df['bureau_average_of_past_loans_per_type'] =  self.df['bureau_number_of_past_loans'] / self.df['bureau_number_of_loan_types']
        self.df['bureau_debt_credit_ratio'] = self.df['bureau_total_customer_debt'] / self.df['bureau_total_customer_credit']
        self.df['bureau_overdue_debt_ratio'] = self.df['bureau_total_customer_overdue'] / self.df['bureau_total_customer_debt']

        del g
        gc.collect()
