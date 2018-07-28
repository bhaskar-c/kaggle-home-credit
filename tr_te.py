import pandas as pd
import gc
import re
from utils import *


class TrainTest:

  def __init__(self, debug=False):
    tr = reduce_mem_usage(read_data('application_train', debug=debug))
    te = reduce_mem_usage(read_data('application_test', debug=debug))
    self.tr_te = tr.append(te, sort=False).reset_index()
    self.df = pd.DataFrame()
    self.df['SK_ID_CURR'] = self.tr_te['SK_ID_CURR']
    self.df['TARGET'] = self.tr_te['TARGET']
    del (tr,te)
    gc.collect()

  def execute(self):
    self.clean()
    self.handpicked()
    self.handcrafted()
    self.aggregations()
    print('Train Test Shape', self.df.shape)
    return self.df

  def clean(self):
    self.tr_te['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace=True)
    self.tr_te['CODE_GENDER'].replace('XNA',np.nan, inplace=True)
    self.tr_te['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    self.tr_te['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace=True)
    self.tr_te['NAME_FAMILY_STATUS'].replace('Unknown', np.nan, inplace=True)
    self.tr_te['ORGANIZATION_TYPE'].replace('XNA', np.nan, inplace=True)


  def handpicked(self):
    self.df['DAYS_BIRTH'] = self.tr_te['DAYS_BIRTH']
    self.df['EXT_SOURCE_3'] = self.tr_te['EXT_SOURCE_3']
    self.df['EXT_SOURCE_2'] = self.tr_te['EXT_SOURCE_2']
    self.df['EXT_SOURCE_1'] = self.tr_te['EXT_SOURCE_1']
    self.df['NEW_IS_DAYS_EMPLOYED_365243'] = (self.tr_te['DAYS_EMPLOYED']== np.nan) # DO WE SEGRAGATE THIS INTO PENSIONERS AND unemployed later ?
    self.df['DAYS_EMPLOYED'] = pd.cut(self.tr_te['DAYS_EMPLOYED'], [-9999999, -7500, -5000, -4500,-4000, -3500, -3000, -2500, -2000, -1500, -1000, -500, 0, 9999999999], labels=[x for x in range(13)])
    self.df['AMT_CREDIT'] = self.tr_te['AMT_CREDIT']
    self.df['AMT_ANNUITY'] = self.tr_te['AMT_ANNUITY']
    self.df['DAYS_ID_PUBLISH'] = self.tr_te['DAYS_ID_PUBLISH']
    self.df['DAYS_REGISTRATION'] = self.tr_te['DAYS_REGISTRATION']
    self.df['REGION_POPULATION_RELATIVE'] = self.tr_te['REGION_POPULATION_RELATIVE'] # kde shows little relation to default
    self.df['AMT_GOODS_PRICE'] = self.tr_te['AMT_GOODS_PRICE'] # - not much of a relationship seen
    self.df['CODE_GENDER'] = self.tr_te['CODE_GENDER']
    self.df['TOTALAREA_MODE'] =  self.tr_te['TOTALAREA_MODE']
    self.df['AMT_INCOME_TOTAL'] = self.tr_te['AMT_INCOME_TOTAL']
    self.df['BASEMENTAREA_AVG'] = self.tr_te['BASEMENTAREA_AVG']
    self.df['LIVINGAREA_MODE'] = self.tr_te['LIVINGAREA_MODE']
    self.df['AMT_REQ_CREDIT_BUREAU_YEAR'] = self.tr_te['AMT_REQ_CREDIT_BUREAU_YEAR']
    self.df['YEARS_BEGINEXPLUATATION_AVG'] = self.tr_te['YEARS_BEGINEXPLUATATION_AVG']
    self.df['LANDAREA_MODE'] = self.tr_te['LANDAREA_MODE']
    self.df['OBS_60_CNT_SOCIAL_CIRCLE'] = self.tr_te['OBS_60_CNT_SOCIAL_CIRCLE']
    self.df['NONLIVINGAREA_AVG'] = self.tr_te['NONLIVINGAREA_AVG']
    self.df['DAYS_LAST_PHONE_CHANGE'] = self.tr_te['DAYS_LAST_PHONE_CHANGE']
    self.df['OWN_CAR_AGE'] = self.tr_te['OWN_CAR_AGE']


  def handcrafted(self):
    self.df['CNT_NON_CHILD']  = self.tr_te['CNT_FAM_MEMBERS'] - self.tr_te['CNT_CHILDREN']

    self.df['AMT_ANNUITY/AMT_INCOME_TOTAL'] = self.tr_te['AMT_ANNUITY'] / self.tr_te['AMT_INCOME_TOTAL']
    self.df['AMT_ANNUITY/AMT_CREDIT'] = self.tr_te['AMT_ANNUITY'] / self.tr_te['AMT_CREDIT']

    self.df['OWN_CAR_AGE/DAYS_BIRTH'] = self.tr_te['OWN_CAR_AGE'] / self.tr_te['DAYS_BIRTH']
    self.df['OWN_CAR_AGE/DAYS_EMPLOYED'] = self.tr_te['OWN_CAR_AGE'] / self.tr_te['DAYS_EMPLOYED']

    self.df['AMT_CREDIT/AMT_ANNUITY'] = self.tr_te['AMT_CREDIT']/(1+self.tr_te['AMT_ANNUITY'])
    self.df['AMT_CREDIT/AMT_GOODS_PRICE'] = self.tr_te['AMT_CREDIT'] / self.tr_te['AMT_GOODS_PRICE']
    self.df['AMT_CREDIT/AMT_INCOME_TOTAL'] = self.tr_te['AMT_CREDIT'] / self.tr_te['AMT_INCOME_TOTAL']
    self.df['AMT_CREDIT/DAYS_BIRTH'] = self.tr_te['AMT_CREDIT'] / (self.tr_te['DAYS_BIRTH']-1)
    self.df['AMT_CREDIT/CNT_FAM_MEMBERS'] = self.tr_te['AMT_CREDIT'] / self.tr_te['CNT_FAM_MEMBERS']
    self.df['AMT_CREDIT/CNT_CHILDREN'] = self.tr_te['AMT_CREDIT'] / (1 + self.tr_te['CNT_CHILDREN'])
    self.df['AMT_CREDIT/CNT_NON_CHILD'] = self.tr_te['AMT_CREDIT'] / self.df['CNT_NON_CHILD']

    self.df['AMT_INCOME_TOTAL/AMT_CREDIT'] = self.tr_te['AMT_INCOME_TOTAL'] / self.tr_te['AMT_CREDIT']
    self.df['AMT_INCOME_TOTAL/CNT_CHILDREN'] = self.tr_te['AMT_INCOME_TOTAL'] / (1 + self.tr_te['CNT_CHILDREN'])
    self.df['AMT_INCOME_TOTAL/CNT_FAM_MEMBERS'] = self.tr_te['AMT_INCOME_TOTAL'] / self.tr_te['CNT_FAM_MEMBERS']
    self.df['AMT_INCOME_TOTAL/CNT_NON_CHILD'] = self.tr_te['AMT_INCOME_TOTAL'] / self.df['CNT_NON_CHILD']
    self.df['AMT_INCOME_TOTAL/CNT_FAM_MEMBERS*AMT_CREDIT'] = (self.tr_te['AMT_INCOME_TOTAL'] / (self.tr_te['CNT_FAM_MEMBERS']*self.tr_te['AMT_CREDIT']))
    self.df['AMT_INCOME_TOTAL/DAYS_BIRTH'] = self.tr_te['AMT_INCOME_TOTAL'] / (self.tr_te['DAYS_BIRTH']+1)

    self.df['DAYS_EMPLOYED/DAYS_BIRTH'] = self.tr_te['DAYS_EMPLOYED'] / self.tr_te['DAYS_BIRTH']
    self.df['DAYS_LAST_PHONE_CHANGE/DAYS_BIRTH'] = self.tr_te['DAYS_LAST_PHONE_CHANGE'] / self.tr_te['DAYS_BIRTH']
    self.df['DAYS_LAST_PHONE_CHANGE/DAYS_EMPLOYED'] = self.tr_te['DAYS_LAST_PHONE_CHANGE'] / self.tr_te['DAYS_EMPLOYED']

    self.df['EXT_SOURCES_PROD'] = self.tr_te['EXT_SOURCE_1'] * self.tr_te['EXT_SOURCE_2'] * self.tr_te['EXT_SOURCE_3']
    self.df['EXT_SOURCE_WEIGHTED'] = self.tr_te.EXT_SOURCE_1 * 2 + self.tr_te.EXT_SOURCE_2 * 3 + self.tr_te.EXT_SOURCE_3 * 4
    self.df['EXT_SCORES_STD'] = self.df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    for function_name in ['min', 'max', 'sum', 'mean', 'nanmedian']:
        self.df['EXT_SOURCES_{}'.format(function_name)] = eval('np.{}'.format(function_name))(self.tr_te[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1)

    self.df['CNT_CHILDREN/CNT_NON_CHILD'] = self.tr_te['CNT_CHILDREN'] / self.df['CNT_NON_CHILD']
    self.df['CNT_CHILDREN/CNT_FAM_MEMBERS'] = self.tr_te['CNT_CHILDREN'] / self.tr_te['CNT_FAM_MEMBERS']
    self.df['NEWLY_EMPLOYED'] = (self.tr_te['DAYS_EMPLOYED'] < -2000).astype(int)
    self.df['YOUNG_AGED'] = (self.tr_te['DAYS_BIRTH'] < -14000).astype(int)
    docs = [_f for _f in self.tr_te.columns if 'FLAG_DOC' in _f]
    self.df['NEW_DOC_IND_KURT'] = self.tr_te[docs].kurtosis(axis=1)
    self.df['NEW_NUMBER_OF_DOCUMENTS_SUBMITTED'] = self.tr_te[docs].sum(axis=1)
    self.df["NEW_DOC_IND_KURT"] = (self.tr_te[[col for col in self.tr_te.columns if re.search(r"FLAG_DOCUMENT", col)]].kurtosis(axis=1))
    self.df["NEW_LIVE_IND_SUM"] = (self.tr_te[["FLAG_OWN_CAR", "FLAG_OWN_REALTY"]].sum(axis=1))
    self.df["NEW_LIVE_IND_KURT"] = (self.tr_te[["FLAG_OWN_CAR", "FLAG_OWN_REALTY"]].kurtosis(axis=1))
    self.df["NEW_CONTACT_IND_SUM"] = (self.tr_te[["FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL"]].sum(axis=1))
    self.df["NEW_CONTACT_IND_KURT"] = (self.tr_te[["FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL"]].kurtosis(axis=1))

    self.df["NEW_REG_IND_SUM"] = (self.tr_te[["REG_REGION_NOT_LIVE_REGION", "REG_REGION_NOT_WORK_REGION", "LIVE_REGION_NOT_WORK_REGION",
     "REG_CITY_NOT_LIVE_CITY", "REG_CITY_NOT_WORK_CITY", "LIVE_CITY_NOT_WORK_CITY"]].sum(axis=1))
    self.df["NEW_REG_IND_KURT"] = (self.tr_te[["REG_REGION_NOT_LIVE_REGION", "REG_REGION_NOT_WORK_REGION", "LIVE_REGION_NOT_WORK_REGION",
     "REG_CITY_NOT_LIVE_CITY", "REG_CITY_NOT_WORK_CITY", "LIVE_CITY_NOT_WORK_CITY"]].kurtosis(axis=1))




  def aggregations(self):
    df = self.tr_te[6:9]
    return df
