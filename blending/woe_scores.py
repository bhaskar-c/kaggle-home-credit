#FINALIZED
import pandas as pd
import math

debug = False

def read_csv_data(file_name, debug, server=True, num_rows=200):
    if server:
        path = '/home/science/data/'
        df = pd.read_hdf(path + file_name + '.h5', 'data')
    else:
        path = '/home/gublu/Desktop/THINKSTATS/Competition/data/'
        if debug:
            df = pd.read_csv(path + file_name + '.csv', nrows=num_rows)
        else:
            df = pd.read_csv(path + file_name + '.csv')
    for col in list(df):
        if str(df[col].dtype) == 'category':
            df[col] = df[col].astype('object')
    return df


class WOEScores:

  def __init__(self,debug):
    self.df = read_csv_data('shiv', debug=debug)
    self.ids = self.df['SK_ID_CURR']
    self.df = self.df.drop('SK_ID_CURR', 1)
    self.woe_df = pd.DataFrame()
    self.bin_the_df()
    self.woe_encoded_df()
    self.save_woe_scores()

  def save_woe_scores(self):
    self.split_cols_tablewise_and_calculate_int_score()
    return (self.df, self.woe_df)


  def split_cols_tablewise_and_calculate_int_score(self):
    b_cols = [col for col in list(self.woe_df) if col.startswith('B_')]
    cc_cols = [col for col in list(self.woe_df) if col.startswith('CC_')]
    ip_cols = [col for col in list(self.woe_df) if col.startswith('IP_')]
    pa_cols = [col for col in list(self.woe_df) if col.startswith('PA_')]
    pos_cols = [col for col in list(self.woe_df) if col.startswith('POS_')]
    all_but_tr_te_cols = b_cols + cc_cols + ip_cols + pa_cols + pos_cols
    tr_te_cols = list(set(list(self.woe_df)) - set(all_but_tr_te_cols))

    tr_te = pd.DataFrame()
    tr_te['SK_ID_CURR'] = self.ids
    tr_te['WOE_SCORE_B']  = self.woe_df[b_cols].sum(axis=1)
    tr_te['WOE_SCORE_CC']  = self.woe_df[cc_cols].sum(axis=1)
    tr_te['WOE_SCORE_IP']  = self.woe_df[ip_cols].sum(axis=1)
    tr_te['WOE_SCORE_PA']  = self.woe_df[pa_cols].sum(axis=1)
    tr_te['WOE_SCORE_POS']  = self.woe_df[pos_cols].sum(axis=1)
    tr_te['WOE_SCORE_ALL_BUT_TR_TE']  = self.woe_df[all_but_tr_te_cols].sum(axis=1)
    tr_te['WOE_SCORE_TR_TE']  = self.woe_df[tr_te_cols].sum(axis=1)
    tr_te.to_csv('woe_tr_te.csv', index= False)

  def bin_the_df(self):
    categorical_columns = [col for col in list(self.df) if self.df[col].nunique()<=20]
    continuous_columns = [col for col in list(self.df) if self.df[col].nunique()>20]
    for col in continuous_columns:
      z = self.df[col].copy(deep=True)
      try:
        binned = pd.cut(z, 20)
      except ValueError as e:
        continue
      self.woe_df[col] = binned
    self.woe_df = pd.concat([self.woe_df, self.df[categorical_columns]], axis=1)
    self.woe_df['TARGET'] = self.df['TARGET']


  def woe_encoded_df(self):
    x_total_good = (self.woe_df['TARGET'] == 0).sum()
    x_total_bad = (self.woe_df['TARGET'] == 1).sum()
    for col in list(self.woe_df):
        if col == 'TARGET':
            continue
        bin_woes = []
        bin_labels = self.woe_df[col].unique()
        print(col)
        for bin_label in bin_labels:
            total = (self.woe_df[col] == bin_label).sum()
            good = ((self.woe_df[col] == bin_label) & (self.woe_df['TARGET'] == 0)).sum()
            good_percent = max(good / x_total_good, 1e-100) # to avoid 0 in woe calculation
            bad = ((self.woe_df[col] == bin_label) & (self.woe_df['TARGET'] == 1)).sum()
            bad_percent = max(bad /x_total_bad, 1e-100) # to avoid inf in woe calculation
            woe = round(math.log(good_percent/bad_percent), 5)
            bin_woes.append(woe)
        self.woe_df[col].replace(dict(zip(bin_labels, bin_woes)), inplace= True)


if __name__ == "__main__":
    WOEScores(debug=debug)


