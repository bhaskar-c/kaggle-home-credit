import pandas as pd
import gc
from utils import *


class CreditCard:

  def __init__(self, debug=False):
    self.cc = reduce_mem_usage(read_data('credit_card_balance', debug=debug))
    self.df = pd.DataFrame({'SK_ID_CURR': self.cc['SK_ID_CURR'].unique()})


  def execute(self):
    self.clean()
    self.static_features()
    self.dynamic_features()
    for col in list(self.df):
      if col != 'SK_ID_CURR':
        self.df.rename(columns={col: 'CC_'+ col}, inplace=True)
    del self.cc
    gc.collect()
    print('Credit Card Shape', self.df.shape)
    return self.df

  def clean(self):
    self.cc['AMT_DRAWINGS_ATM_CURRENT'][self.cc['AMT_DRAWINGS_ATM_CURRENT'] < 0] = np.nan
    self.cc['AMT_DRAWINGS_CURRENT'][self.cc['AMT_DRAWINGS_CURRENT'] < 0] = np.nan


  def static_features(self):
    self.cc['number_of_installments'] = self.cc.groupby(by=['SK_ID_CURR', 'SK_ID_PREV'])['CNT_INSTALMENT_MATURE_CUM'].agg('max').reset_index()['CNT_INSTALMENT_MATURE_CUM']
    self.cc['credit_card_max_loading_of_credit_limit'] = self.cc.groupby(by=['SK_ID_CURR', 'SK_ID_PREV', 'AMT_CREDIT_LIMIT_ACTUAL']).apply(lambda x: x.AMT_BALANCE.max() / x.AMT_CREDIT_LIMIT_ACTUAL.max()).reset_index()[0]
    groupby = self.cc.groupby(by=['SK_ID_CURR'])

    g = groupby['SK_ID_PREV'].agg('nunique').reset_index()
    g.rename(index=str, columns={'SK_ID_PREV': 'credit_card_number_of_loans'}, inplace=True)
    self.df = self.df.merge(g, on=['SK_ID_CURR'], how='left')

    g = groupby['SK_DPD'].agg('mean').reset_index()
    g.rename(index=str, columns={'SK_DPD': 'credit_card_average_of_days_past_due'}, inplace=True)
    self.df = self.df.merge(g, on=['SK_ID_CURR'], how='left')

    g = groupby['AMT_DRAWINGS_ATM_CURRENT'].agg('sum').reset_index()
    g.rename(index=str, columns={'AMT_DRAWINGS_ATM_CURRENT': 'credit_card_drawings_atm'}, inplace=True)
    self.df = self.df.merge(g, on=['SK_ID_CURR'], how='left')

    g = groupby['AMT_DRAWINGS_CURRENT'].agg('sum').reset_index()
    g.rename(index=str, columns={'AMT_DRAWINGS_CURRENT': 'credit_card_drawings_total'}, inplace=True)
    self.df = self.df.merge(g, on=['SK_ID_CURR'], how='left')

    g = groupby['number_of_installments'].agg('sum').reset_index()
    g.rename(index=str, columns={'number_of_installments': 'credit_card_total_installments'}, inplace=True)
    self.df = self.df.merge(g, on=['SK_ID_CURR'], how='left')

    g = groupby['credit_card_max_loading_of_credit_limit'].agg('mean').reset_index()
    g.rename(index=str,
         columns={'credit_card_max_loading_of_credit_limit': 'credit_card_avg_loading_of_credit_limit'},
         inplace=True)
    self.df = self.df.merge(g, on=['SK_ID_CURR'], how='left')

    self.df['credit_card_cash_card_ratio'] = self.df['credit_card_drawings_atm'] / self.df['credit_card_drawings_total']
    self.df['credit_card_installments_per_loan'] = (self.df['credit_card_total_installments'] / self.df['credit_card_number_of_loans'])
    del g
    gc.collect()


  def dynamic_features(self):
    credit_card_sorted = self.cc.sort_values(['SK_ID_CURR', 'MONTHS_BALANCE'])
    groupby = credit_card_sorted.groupby(by=['SK_ID_CURR'])
    credit_card_sorted['credit_card_monthly_diff'] = groupby['AMT_BALANCE'].diff()
    groupby = credit_card_sorted.groupby(by=['SK_ID_CURR'])

    g = groupby['credit_card_monthly_diff'].agg('mean').reset_index()
    self.df = self.df.merge(g, on=['SK_ID_CURR'], how='left')
    del g
    gc.collect()

  def aggregations(self):
    cc_aggregations = {
        'CNT_DRAWINGS_ATM_CURRENT': ['mean', 'var', 'max', 'sum'],
        'AMT_PAYMENT_CURRENT': ['mean'],
        'AMT_CREDIT_LIMIT_ACTUAL': ['mean'],
        'CNT_DRAWINGS_CURRENT': ['var', 'mean', 'sum'],
        'AMT_BALANCE': ['mean', 'max', 'min'],
        'MONTHS_BALANCE': ['mean', 'min', 'var', 'sum'],
        'AMT_DRAWINGS_CURRENT': ['mean', 'max']

    }

    cc_g = self.cc.groupby('SK_ID_CURR').agg(cc_aggregations)
    cc_g.columns = pd.Index([e[0] + "_" + e[1].upper() for e in cc_g.columns.tolist()])
    cc_g['SK_ID_CURR'] = cc_g.index
    cc_g['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()      # Count credit card lines
    self.df = self.df.merge(cc_g, on=['SK_ID_CURR'], how='left')
