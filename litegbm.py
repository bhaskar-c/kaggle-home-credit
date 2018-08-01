from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np
import datetime
import gc
from utils import *

class LiteGBM:

  def __init__(self, df, debug):
    self.debug = False #debug
    #df = df.drop('NEW_LIVE_IND_SUM', 1) # giving dtypes invalid errors in lightgbm
    df = label_encode_it(df)
    df = one_hot_encode_it(df)
    self.train_df = df[df['TARGET'].notnull()]
    self.y = self.train_df['TARGET']
    self.test_df = df[df['TARGET'].isnull()]
    self.test_df = self.test_df.drop('TARGET', 1)
    self.num_folds = 5
    '''self.params = {'device': 'cpu', 'boosting_type': 'gbdt', 'objective': 'binary',
    'metric': 'auc', 'is_unbalance': False, 'scale_pos_weight': 1,
    'learning_rate': 0.02, 'max_bin': 300,
    'max_depth': -1, 'num_leaves': 30, 'min_child_samples': 70, 'subsample': 1.0,
    'colsample_bytree': 0.01, 'subsample_freq': 1, 'min_gain_to_split': 0.5,
    'reg_lambda': 100, 'reg_alpha': 0.0, 'nthread': 1, 'n_estimators': 5000,
    'early_stopping_rounds': 100, 'verbose': 1}'''
    #self.training_params = ['number_boosting_rounds', 'early_stopping_rounds']
    self.params = {  'objective': 'binary', 'metric': 'auc','nthread': 4, #'boost_from_average':False,
            'n_estimators':10000, 'learning_rate':1,
            'max_depth': -1, 'num_leaves':32, 'max_bin': 25,
            'feature_fraction': 0.80, 'bagging_fraction':0.7, 'bagging_freq': 1,
            'reg_alpha':3, 'reg_lambda':3,
            'min_split_gain':0.0000215, #'min_child_weight':40,
            #'class_weight': 'balanced',
            'verbose':-1}
    self.evaluation_function = None
    #self.callbacks = callbacks(channel_prefix=name)
    del df
    gc.collect()
    self.kfold_lightgbm()




  def kfold_lightgbm(self):
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(self.train_df.shape, self.test_df.shape))
    folds = StratifiedKFold(n_splits= self.num_folds, shuffle=True, random_state=1)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(self.train_df.shape[0])
    sub_preds = np.zeros(self.test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in self.train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV']]
    #categorical_features = ['NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','CNT_CHILDREN','NAME_TYPE_SUITE','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','FLAG_MOBIL','FLAG_EMP_PHONE','FLAG_WORK_PHONE','FLAG_CONT_MOBILE','FLAG_PHONE','FLAG_EMAIL','CNT_FAM_MEMBERS','REGION_RATING_CLIENT','REGION_RATING_CLIENT_W_CITY','WEEKDAY_APPR_PROCESS_START','REG_REGION_NOT_LIVE_REGION','REG_REGION_NOT_WORK_REGION','LIVE_REGION_NOT_WORK_REGION','REG_CITY_NOT_LIVE_CITY','REG_CITY_NOT_WORK_CITY','LIVE_CITY_NOT_WORK_CITY','FONDKAPREMONT_MODE','HOUSETYPE_MODE','WALLSMATERIAL_MODE','EMERGENCYSTATE_MODE','DEF_30_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE','FLAG_DOCUMENT_3','FLAG_DOCUMENT_5','FLAG_DOCUMENT_6','FLAG_DOCUMENT_8','FLAG_DOCUMENT_9','FLAG_DOCUMENT_11','FLAG_DOCUMENT_13','FLAG_DOCUMENT_14','FLAG_DOCUMENT_15','FLAG_DOCUMENT_16','FLAG_DOCUMENT_18','FLAG_DOCUMENT_19','FLAG_DOCUMENT_20','AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_QRT','be_credit_status_sold_count','be_CREDIT_CURRENCY','be_last_CREDIT_TYPE','bb_b_latest_month_STATUS_DPD_count','be_sum_CNT_CREDIT_PROLONG','be_is_AMT_CREDIT_MAX_OVERDUE','be_is_AMT_CREDIT_SUM_OVERDUE','pae_NAME_CONTRACT_TYPE','pae_num_of_previous_applications_unused_offers','pae_name_client_type_new_count','pae_name_client_type_xna_count','pae_NAME_PAYMENT_TYPE','pae_NAME_YIELD_GROUP','pae_CODE_REJECT_REASON','pae_NAME_CLIENT_TYPE','pae_NAME_TYPE_SUITE','pae_NAME_PORTFOLIO','pae_NAME_PRODUCT_TYPE','pae_CHANNEL_TYPE','pae_NAME_SELLER_INDUSTRY','pae_PRODUCT_COMBINATION','pos_pa_latest_NAME_CONTRACT_STATUS_y','does_client_have_a_credit_card_with_us','does_client_have_pos_cas_balance_record_with_us','total_number_of_documents_submitted','is_nan_EXT_SOURCE_1','is_nan_OCCUPATION_TYPE','is_nan_AMT_REQ_CREDIT_BUREAU_DAY','is_nan_DAYS_EMPLOYED']
    #indexes_of_categories = [X_train.columns.get_loc(col) for col in categorical_features]
    #sample_weight = np.array([11 if i == 1 else 1 for i in y_train])
    #sample_weight.shape
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(self.train_df[feats], self.train_df['TARGET'])):
      train_x, train_y = self.train_df[feats].iloc[train_idx], self.train_df['TARGET'].iloc[train_idx]
      valid_x, valid_y = self.train_df[feats].iloc[valid_idx], self.train_df['TARGET'].iloc[valid_idx]
      clf = LGBMClassifier(**self.params)
      clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], eval_metric= 'auc', verbose= 100, early_stopping_rounds= 200 )
      oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
      sub_preds += clf.predict_proba(self.test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits
      fold_importance_df = pd.DataFrame()
      fold_importance_df["feature"] = feats
      fold_importance_df["importance"] = clf.feature_importances_
      fold_importance_df["fold"] = n_fold + 1
      feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
      print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
      del clf, train_x, train_y, valid_x, valid_y
      gc.collect()
    full_auc = roc_auc_score(self.train_df['TARGET'], oof_preds)
    print('Full AUC score %.6f' % full_auc)
    # Write submission file and plot feature importance
    if not self.debug:
      self.test_df['TARGET'] = sub_preds
      submission = self.test_df[['SK_ID_CURR', 'TARGET']]
      self.save_k_fold_results(feature_importance_df, full_auc, params, submission)

  def save_k_fold_results(self, feature_importance_df_, full_auc, params, submission):
      ts = datetime.datetime.now().strftime("%b%d,%I:%M")
      cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False).index
      best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]

      plt.figure(figsize=(8, 10))
      final = best_features.sort_values(by="importance", ascending=False)
      sns.barplot(x="importance", y="feature", data=final[:30])
      plt.title('LightGBM Features (avg over folds)')
      plt.tight_layout()

      final.drop_duplicates(subset='feature', inplace=True)


      plt.savefig('results/plots/kf_' + ts +'_auc_' + str(full_auc) +'.png')
      final.to_csv('results/importance/kf_' + ts +'_auc_' + str(full_auc) +'.csv', index=False)
      submission.to_csv('results/submissions/kf_' + ts +'_auc_' + str(full_auc) +'.csv', index= False)
      write_to_file(params, full_auc)
