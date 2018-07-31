import pandas as pd
import gc
from utils import *


class PreviousApplication:

  def __init__(self, debug=False):
    self.pa = reduce_mem_usage(read_data('previous_application', debug=debug))
    self.df = pd.DataFrame({'SK_ID_CURR': self.pa['SK_ID_CURR'].unique()})


  def execute(self):
    self.clean()
    self.handcrafted()
    self.aggregations()
    for col in list(self.df):
      if col != 'SK_ID_CURR':
        self.df.rename(columns={col: 'PA_'+ col}, inplace=True)
    del self.pa
    gc.collect()
    print('Previous Application Shape', self.df.shape)
    return self.df

  def clean(self):
    self.pa['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    self.pa['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    self.pa['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    self.pa['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    self.pa['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)

  def handcrafted(self):
        self.pa = self.pa.sort_values(['SK_ID_CURR', 'DAYS_DECISION'])
        prev_app_sorted_groupby = self.pa.groupby(by=['SK_ID_CURR'])

        self.pa['WAS_APPROVED'] = (self.pa['NAME_CONTRACT_STATUS'] == 'Approved').astype('int')
        g = prev_app_sorted_groupby['WAS_APPROVED'].last().reset_index()
        self.df = self.df.merge(g, on=['SK_ID_CURR'], how='left')

        self.pa['WAS_REFUSED'] = (self.pa['NAME_CONTRACT_STATUS'] == 'Refused').astype('int')
        g = prev_app_sorted_groupby['WAS_REFUSED'].last().reset_index()
        self.df = self.df.merge(g, on=['SK_ID_CURR'], how='left')

        g = prev_app_sorted_groupby['SK_ID_PREV'].agg('nunique').reset_index()
        g.rename(index=str, columns={'SK_ID_PREV': 'NUM_PREVIOUS_APPLICATIONS'}, inplace=True)
        self.df = self.df.merge(g, on=['SK_ID_CURR'], how='left')

        g = self.pa.groupby(by=['SK_ID_CURR'])['WAS_REFUSED'].mean().reset_index()
        g.rename(index=str, columns={'WAS_REFUSED': 'FRACTION_REFUSED_APPLICATION'},  inplace=True)
        self.df = self.df.merge(g, on=['SK_ID_CURR'], how='left')

        self.pa['WAS_REVOLVING_LOAN'] = (self.pa['NAME_CONTRACT_TYPE'] == 'Revolving loans').astype('int')
        g = self.pa.groupby(by=['SK_ID_CURR'])['WAS_REVOLVING_LOAN'].last().reset_index()
        self.df = self.df.merge(g, on=['SK_ID_CURR'], how='left')

        for number in [1, 2, 3, 4, 5]:
            self.pa_tail = prev_app_sorted_groupby.tail(number)
            tail_groupby = self.pa_tail.groupby(by=['SK_ID_CURR'])
            g = tail_groupby['CNT_PAYMENT'].agg('mean').reset_index()
            g.rename(index=str, columns={'CNT_PAYMENT': 'TERM_OF_LAST_{}_CREDITS_mean'.format(number)}, inplace=True)
            self.df = self.df.merge(g, on=['SK_ID_CURR'], how='left')

            g = tail_groupby['DAYS_DECISION'].agg('mean').reset_index()
            g.rename(index=str, columns={'DAYS_DECISION': 'DAYS_DECISION_LAST_{}_CREDITS_mean'.format(number)}, inplace=True)
            self.df = self.df.merge(g, on=['SK_ID_CURR'], how='left')

            g = tail_groupby['DAYS_FIRST_DRAWING'].agg('mean').reset_index()
            g.rename(index=str, columns={'DAYS_FIRST_DRAWING': 'DAYS_FIRST_DRAWINGS_LAST_{}_CREDITS_mean'.format(number)}, inplace=True)
            self.df = self.df.merge(g, on=['SK_ID_CURR'], how='left')




  def aggregations(self):
    self.pa['AMT_APPLICATION/AMT_CREDIT'] = self.pa['AMT_APPLICATION'] / self.pa['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': [ 'max', 'mean'],
        'AMT_APPLICATION': [ 'max','mean'],
        'AMT_CREDIT': [ 'max', 'mean'],
        'AMT_APPLICATION/AMT_CREDIT': [ 'max', 'mean'],
        'AMT_DOWN_PAYMENT': [ 'max', 'mean'],
        'AMT_GOODS_PRICE': [ 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': [ 'max', 'mean'],
        'RATE_DOWN_PAYMENT': [ 'max', 'mean'],
        'DAYS_DECISION': [ 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    prev_agg = self.pa.groupby('SK_ID_CURR').agg({**num_aggregations})
    prev_agg.columns = pd.Index([ e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = self.pa[self.pa['NAME_CONTRACT_STATUS'] == 'Approved']
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = self.pa[self.pa['NAME_CONTRACT_STATUS'] == 'Refused']
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    self.df = self.df.merge(prev_agg, on=['SK_ID_CURR'], how='left')
    del approved, approved_agg, refused, refused_agg
    gc.collect()

