import pandas as pd
import gc
from functools import partial
from utils import *


class InstallmentPayments:

  def __init__(self, debug=False):
    self.ip = reduce_mem_usage(read_data('installments_payments', debug=debug))
    self.df = pd.DataFrame({'SK_ID_CURR': self.ip['SK_ID_CURR'].unique()})
    self.last_k_agg_periods = [1, 5, 10, 20, 50, 100]
    self.last_k_agg_period_fractions = [(5, 20), (5, 50), (10, 50), (10, 100), (20, 100)]
    self.last_k_trend_periods = [10, 50, 100, 500]
    self.num_workers = 1

  def execute(self):
    self.clean()
    self.preprocess()
    self.handcrafted()     # this alone creates 239 features
    self.aggregations()

    del self.ip
    gc.collect()
    for col in list(self.df):
      if col != 'SK_ID_CURR':
        self.df.rename(columns={col: 'IP_'+ col}, inplace=True)
    print('Installment Shape', self.df.shape)
    return self.df

  def preprocess(self):
    self.ip['DPD'] = self.ip['DAYS_ENTRY_PAYMENT'] - self.ip['DAYS_INSTALMENT'] # installment_paid_late_in_days
    self.ip['DBD'] = self.ip['DAYS_INSTALMENT'] - self.ip['DAYS_ENTRY_PAYMENT']
    self.ip['DPD'] = self.ip['DPD'].apply(lambda x: x if x > 0 else 0)
    self.ip['DBD'] = self.ip['DBD'].apply(lambda x: x if x > 0 else 0)
    self.ip['AMT_PAYMENT/AMT_INSTALMENT'] = self.ip['AMT_PAYMENT'] / (1+self.ip['AMT_INSTALMENT'])
    self.ip['AMT_PAYMENT-AMT_INSTALMENT'] = self.ip['AMT_PAYMENT'] - self.ip['AMT_INSTALMENT'] #installment_paid_over_amount
    self.ip['IS_DPD'] = (self.ip['DPD'] > 0).astype(int) # installment_paid_late
    self.ip['AMT_PAYMENT>AMT_INSTALMENT'] = (self.ip['AMT_PAYMENT-AMT_INSTALMENT'] > 0).astype(int) # installment_paid_over

  def clean(self):
    pass

  def handcrafted(self):
    groupby = self.ip.groupby(['SK_ID_CURR'])
    func = partial(InstallmentPayments.generate_features,
                 agg_periods=self.last_k_agg_periods,
                 period_fractions=self.last_k_agg_period_fractions,
                 trend_periods=self.last_k_trend_periods)
    features = parallel_apply(groupby, func, index_name='SK_ID_CURR', num_workers=self.num_workers).reset_index()
    self.df = self.df.merge(features, on='SK_ID_CURR', how='left')


  def aggregations(self):
    aggregations = {
        'DAYS_ENTRY_PAYMENT': ['min', 'max', 'mean', 'sum', 'std'],
        'AMT_INSTALMENT':['min', 'sum', 'mean', 'max', 'std'],
        'DBD': ['sum', 'std', 'mean', 'max', 'min'],
        'AMT_PAYMENT':['sum', 'min', 'mean', 'std', 'max'],
        'DPD': ['mean', 'std', 'sum'],
        'AMT_PAYMENT/AMT_INSTALMENT': ['mean'],
        'AMT_PAYMENT-AMT_INSTALMENT': ['mean', 'max', 'var'],
        'DPD': ['mean', 'max'],
        'SK_ID_PREV': ['nunique'],
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'NUM_INSTALMENT_NUMBER': ['count','max'],
        'DAYS_INSTALMENT': ['min', 'var'],
    }
    ip_g = self.ip.groupby('SK_ID_CURR').agg(aggregations)
    ip_g.columns = pd.Index([e[0] + "_" + e[1].upper() for e in ip_g.columns.tolist()])
    ip_g['SK_ID_CURR'] = ip_g.index
    self.df = self.df.merge(ip_g, on=['SK_ID_CURR'], how='left')



  @staticmethod
  def generate_features(gr, agg_periods, trend_periods, period_fractions):
    all_installment_features = InstallmentPayments.all_installment_features(gr)
    agg = InstallmentPayments.last_k_installment_features_with_fractions(gr, agg_periods, period_fractions)
    trend = InstallmentPayments.trend_in_last_k_installment_features(gr, trend_periods)
    last = InstallmentPayments.last_loan_features(gr)
    features = {**all_installment_features, **agg, **trend, **last}
    return pd.Series(features)

  @staticmethod
  def all_installment_features(gr):
    return InstallmentPayments.last_k_installment_features(gr, periods=[10e16])

  @staticmethod
  def last_k_installment_features_with_fractions(gr, periods, period_fractions):
    features = InstallmentPayments.last_k_installment_features(gr, periods)
    for short_period, long_period in period_fractions:
      short_feature_names = get_feature_names_by_period(features, short_period)
      long_feature_names = get_feature_names_by_period(features, long_period)
    for short_feature, long_feature in zip(short_feature_names, long_feature_names):
      old_name_chunk = '_{}_'.format(short_period)
      new_name_chunk = '_{}by{}_fraction_'.format(short_period, long_period)
      fraction_feature_name = short_feature.replace(old_name_chunk, new_name_chunk)
      features[fraction_feature_name] = safe_div(features[short_feature], features[long_feature])
    return features

  @staticmethod
  def last_k_installment_features(gr, periods):
    gr_ = gr.copy()
    gr_.sort_values(['DAYS_INSTALMENT'], ascending=False, inplace=True)
    features = {}
    for period in periods:
      if period > 10e10:
        period_name = 'all_installment_'
        gr_period = gr_.copy()
      else:
        period_name = 'last_{}_'.format(period)
        gr_period = gr_.iloc[:period]
        features = add_features_in_group(features, gr_period, 'NUM_INSTALMENT_VERSION', ['sum', 'mean', 'max', 'min', 'std', 'median', 'skew', 'kurt', 'iqr'], period_name)
        features = add_features_in_group(features, gr_period, 'DPD', ['sum', 'mean', 'max', 'min', 'std', 'median', 'skew', 'kurt', 'iqr'], period_name)
        features = add_features_in_group(features, gr_period, 'IS_DPD', ['count', 'mean'], period_name)
        features = add_features_in_group(features, gr_period, 'AMT_PAYMENT-AMT_INSTALMENT', ['sum', 'mean', 'max', 'min', 'std', 'median', 'skew', 'kurt', 'iqr'], period_name)
        features = add_features_in_group(features, gr_period, 'AMT_PAYMENT>AMT_INSTALMENT', ['count', 'mean'], period_name)
    return features

  @staticmethod
  def trend_in_last_k_installment_features(gr, periods):
    gr_ = gr.copy()
    gr_.sort_values(['DAYS_INSTALMENT'], ascending=False, inplace=True)
    features = {}
    for period in periods:
      gr_period = gr_.iloc[:period]
      features = add_trend_feature(features, gr_period, 'DPD', '{}_period_trend_'.format(period))
      features = add_trend_feature(features, gr_period, 'AMT_PAYMENT-AMT_INSTALMENT', '{}_period_trend_'.format(period))
    return features

  @staticmethod
  def last_loan_features(gr):
    gr_ = gr.copy()
    gr_.sort_values(['DAYS_INSTALMENT'], ascending=False, inplace=True)
    last_installment_id = gr_['SK_ID_PREV'].iloc[0]
    gr_ = gr_[gr_['SK_ID_PREV'] == last_installment_id]
    features = {}
    features = add_features_in_group(features, gr_, 'DPD', ['sum', 'mean', 'max', 'min', 'std'], 'last_loan_')
    features = add_features_in_group(features, gr_, 'IS_DPD', ['count', 'mean'], 'last_loan_')
    features = add_features_in_group(features, gr_, 'AMT_PAYMENT-AMT_INSTALMENT', ['sum', 'mean', 'max', 'min', 'std'], 'last_loan_')
    features = add_features_in_group(features, gr_, 'AMT_PAYMENT>AMT_INSTALMENT', ['count', 'mean'], 'last_loan_')
    return features
