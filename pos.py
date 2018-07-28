import pandas as pd
import gc
from utils import *


class POSCASHBalance:

  def __init__(self, debug=False):
    self.pos = reduce_mem_usage(read_data('POS_CASH_balance', debug=debug))
    self.df = pd.DataFrame({'SK_ID_CURR': self.pos['SK_ID_CURR'].unique()})


  def execute(self):
    self.clean()
    self.handcrafted()
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

  def handcrafted(self):
    pass

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

