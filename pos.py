import pandas as pd
import gc
from functools import partial
from utils import *


class POSCASHBalance:

  def __init__(self, debug=False):
    self.pos = reduce_mem_usage(read_data('POS_CASH_balance', debug=debug))
    self.df = pd.DataFrame({'SK_ID_CURR': self.pos['SK_ID_CURR'].unique()})
    self.last_k_agg_periods = [6,12,30]
    self.last_k_trend_periods = [6,12]
    self.num_workers = 3

  def execute(self):
    self.preprocess()
    self.clean()
    self.handcrafted()  # 46 features
    self.aggregations()
    for col in list(self.df):
      if col != 'SK_ID_CURR':
        self.df.rename(columns={col: 'POS_'+ col}, inplace=True)
    del self.pos
    gc.collect()
    print('POSCash Shape', self.df.shape)
    return self.df

  def clean(self):
    pass

  def preprocess(self):
    self.pos['IS_CONTRACT_STATUS_COMPLETED'] = self.pos['NAME_CONTRACT_STATUS'] == 'Completed'
    self.pos['DPD>0'] = (self.pos['SK_DPD'] > 0).astype(int)
    self.pos['DPDDEF>0'] = (self.pos['SK_DPD_DEF'] > 0).astype(int)


  def handcrafted(self):
    groupby = self.pos.groupby(['SK_ID_CURR'])
    func = partial(POSCASHBalance.generate_features,
                       agg_periods=self.last_k_agg_periods,
                       trend_periods=self.last_k_trend_periods)
    features = parallel_apply(groupby, func, index_name='SK_ID_CURR', num_workers=self.num_workers).reset_index()
    self.df = self.df.merge(features, on='SK_ID_CURR', how='left')

  def aggregations(self):
    count_name_contract_status = self.pos.groupby('SK_ID_CURR').NAME_CONTRACT_STATUS.value_counts().unstack(fill_value=0)
    count_name_contract_status.columns = pd.Index(['NAME_CONTRACT_STATUS_' + str(e).strip() for e in count_name_contract_status.columns.tolist()])
    count_name_contract_status['SK_ID_CURR'] = count_name_contract_status.index
    #count_name_contract_status = count_name_contract_status[['NAME_CONTRACT_STATUS_Active', 'NAME_CONTRACT_STATUS_Amortized debt', 'NAME_CONTRACT_STATUS_Completed', 'NAME_CONTRACT_STATUS_Demand', 'SK_ID_CURR']]
    count_name_contract_status['NAME_CONTRACT_STATUS_Active/Completed'] = round(count_name_contract_status['NAME_CONTRACT_STATUS_Active']/ (0.0001+count_name_contract_status['NAME_CONTRACT_STATUS_Completed']))
    count_name_contract_status = count_name_contract_status.rename(columns=lambda x: x.replace(' ', '_').upper())
    self.df = self.df.merge(count_name_contract_status, on=['SK_ID_CURR'], how='left')

    self.pos['DPD_MINUS_DPD_DEF'] = self.pos.SK_DPD - self.pos.SK_DPD_DEF

    # Features
    pos_aggregations = {
          'SK_ID_PREV': ['nunique'],
          'MONTHS_BALANCE': ['mean', 'size', 'max', 'min', 'var'],
          'SK_DPD': ['mean'],
          'CNT_INSTALMENT': ['first'],
          'DPD_MINUS_DPD_DEF': ['max'],
          'SK_DPD_DEF': ['max', 'mean'],

    }
    pos_g = self.pos.groupby('SK_ID_CURR').agg(pos_aggregations)
    pos_g.columns = pd.Index([e[0] + "_" + e[1].upper() for e in pos_g.columns.tolist()])
    pos_g['SK_ID_CURR'] = pos_g.index
    self.df = self.df.merge(pos_g, on=['SK_ID_CURR'], how='left')

    latest_rows = self.pos.loc[self.pos.groupby(["SK_ID_CURR"])["MONTHS_BALANCE"].idxmax()]
    latest_rows = latest_rows[['SK_ID_CURR', 'SK_DPD_DEF']]
    latest_rows.columns = pd.Index(['LATEST_' + e for e in latest_rows.columns.tolist()])
    latest_rows.rename(columns={'LATEST_SK_ID_CURR':'SK_ID_CURR'}, inplace=True)
    self.df = self.df.merge(latest_rows, on=['SK_ID_CURR'], how='left')

    del (pos_g, latest_rows)
    gc.collect()

  @staticmethod
  def generate_features(gr, agg_periods, trend_periods):
    one_time = POSCASHBalance.one_time_features(gr)
    all_installment_features = POSCASHBalance.all_installment_features(gr)
    agg = POSCASHBalance.last_k_installment_features(gr, agg_periods)
    trend = POSCASHBalance.trend_in_last_k_installment_features(gr, trend_periods)
    last = POSCASHBalance.last_loan_features(gr)
    features = {**one_time, **all_installment_features, **agg, **trend, **last}
    return pd.Series(features)

  @staticmethod
  def one_time_features(gr):
    gr_ = gr.copy()
    gr_.sort_values(['MONTHS_BALANCE'], inplace=True)
    features = {}
    features['CNT_INSTALMENT_FUTURE_tail'] = gr_['CNT_INSTALMENT_FUTURE'].tail(1)
    features['IS_CONTRACT_STATUS_COMPLETED_sum'] = gr_['IS_CONTRACT_STATUS_COMPLETED'].agg('sum')
    return features

  @staticmethod
  def all_installment_features(gr):
    return POSCASHBalance.last_k_installment_features(gr, periods=[10e16])

  @staticmethod
  def last_k_installment_features(gr, periods):
    gr_ = gr.copy()
    gr_.sort_values(['MONTHS_BALANCE'], ascending=False, inplace=True)
    features = {}
    for period in periods:
      if period > 10e10:
        period_name = 'all_installment_'
        gr_period = gr_.copy()
      else:
        period_name = 'last_{}_'.format(period)
        gr_period = gr_.iloc[:period]
      features = add_features_in_group(features, gr_period, 'DPD>0', ['count', 'mean'], period_name)
      features = add_features_in_group(features, gr_period, 'DPDDEF>0', ['count', 'mean'], period_name)
      features = add_features_in_group(features, gr_period, 'SK_DPD', ['sum', 'mean', 'max', 'std', 'skew', 'kurt'], period_name)
      features = add_features_in_group(features, gr_period, 'SK_DPD_DEF', ['sum', 'mean', 'max', 'std', 'skew', 'kurt'], period_name)
      return features

  @staticmethod
  def trend_in_last_k_installment_features(gr, periods):
    gr_ = gr.copy()
    gr_.sort_values(['MONTHS_BALANCE'], ascending=False, inplace=True)
    features = {}
    for period in periods:
      gr_period = gr_.iloc[:period]
      features = add_trend_feature(features, gr_period, 'SK_DPD', '{}_period_trend_'.format(period))
      features = add_trend_feature(features, gr_period, 'SK_DPD_DEF', '{}_period_trend_'.format(period))
      features = add_trend_feature(features, gr_period, 'CNT_INSTALMENT_FUTURE', '{}_period_trend_'.format(period))
    return features

  @staticmethod
  def last_loan_features(gr):
    gr_ = gr.copy()
    gr_.sort_values(['MONTHS_BALANCE'], ascending=False, inplace=True)
    last_installment_id = gr_['SK_ID_PREV'].iloc[0]
    gr_ = gr_[gr_['SK_ID_PREV'] == last_installment_id]
    features={}
    features = add_features_in_group(features, gr_, 'DPD>0', ['count', 'sum', 'mean'],  'last_loan_')
    features = add_features_in_group(features, gr_, 'DPDDEF>0', ['mean'], 'last_loan_')
    features = add_features_in_group(features, gr_, 'SK_DPD', ['sum', 'mean', 'max', 'std'], 'last_loan_')
    features = add_features_in_group(features, gr_, 'SK_DPD_DEF', ['sum', 'mean', 'max', 'std'], 'last_loan_')
    return features
