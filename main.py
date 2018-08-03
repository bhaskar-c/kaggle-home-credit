from feature_extraction.tr_te import TrainTest
from feature_extraction.b import Bureau
from feature_extraction.cc import CreditCard
from feature_extraction.ip import InstallmentPayments
from feature_extraction.pa import PreviousApplication
from feature_extraction.pos import POSCASHBalance
from litegbm import LiteGBM
from blending.woe_scores import WOEScores
import gc
import warnings
from utils import *
warnings.simplefilter(action='ignore')
#warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

# woe
# auto_encode
# target encoding
# categorical embedding vith neural network
# libfm, vowpal vabbit
# multilayer stacking
# https://www.kaggle.com/kingychiu/dimensionality-reduction-on-application-data
# https://www.kaggle.com/adamsfei/knn-on-application-train-pca-for-ensembling/notebook
# unsupervised methods that reduce dimensionality (like SVD, PCA, ISOMAP, KDTREE clustering)
# similarlity features
# interaction features - like popular product AND good rating
# factorization machines
# https://github.com/fred-navruzov/featuretools-workshop/blob/master/featuretools-workshop.ipynb
# https://www.kaggle.com/kailex/tidy-xgb-all-tables-0-796/code
# genetic programming https://www.kaggle.com/scirpus/hybrid-jeepy-and-lgb-ii
# https://www.kaggle.com/ogrellier/feature-selection-with-null-importances
# https://www.kaggle.com/davidsalazarv95/fast-ai-pytorch-starter-version-two
# https://www.kaggle.com/shep312/deep-learning-in-tf-with-upsampling-lb-758
# https://github.com/turi-code/python-libffm/blob/master/examples/basic.py

#scp root@159.65.83.37:/home/science/data/shiv.h5 /home/gublu/Desktop/THINKSTATS/Competition/hdf


debug = False

def post_process_data(df):
  return df

def change_data(): # imported result of this to /home/gublu/Desktop/THINKSTATS/Competition/hdf/shiv.h5
  df = TrainTest(debug=debug).execute()
  b = Bureau(debug=debug).execute()
  df = df.merge(b, on=['SK_ID_CURR'], how='left')
  cc = CreditCard(debug=debug).execute()
  df = df.merge(cc, on=['SK_ID_CURR'], how='left')
  ip = InstallmentPayments(debug=debug).execute()
  df = df.merge(ip, on=['SK_ID_CURR'], how='left')
  pa = PreviousApplication(debug=debug).execute()
  df = df.merge(pa, on=['SK_ID_CURR'], how='left')
  pos = POSCASHBalance(debug=debug).execute()
  df = df.merge(pos, on=['SK_ID_CURR'], how='left')
  del (b,cc,ip,pa, pos)
  gc.collect()
  df = post_process_data(df)
  #df = reduce_mem_usage(df)
  print(df.shape)
  #df.to_csv('shiv_300.csv', index=False)
  #############df.to_hdf('../hdf/shiv.h5', key='data', mode='w', format="table")
  #df.to_hdf('/home/science/data/shiv.h5', key='data', mode='w', format="table")
  return df


def main():
    #df = change_data()
    #print(df.shape)
    #scaler = StandardScaler()
    #df[['all_num_features here']] = scaler.fit_transform(df[['all_num_features here']])
    #pe(df.shape)
    df = read_hdf_data('shiv', debug)
    df.to_csv('/home/gublu/Desktop/THINKSTATS/Competition/data/shiv.csv', index=False)
    #train_df = read_csv_data('shiv_train', debug)
    #test_df = read_csv_data('shiv_test', debug)

    print(train_df.shape, test_df.shape)
    #dt(df)
    #df = add_as_type_category(df)
    #df =reduce_mem_usage(df)
    #full_train(df, params)
    #train_test_split_run(df, params)
    #LiteGBM(df, debug= debug)
    #play_on_done()

if __name__ == "__main__":
    main()



