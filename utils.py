import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import kurtosis, iqr, skew
from sklearn.preprocessing import LabelEncoder
import multiprocessing as mp
import sys
from tqdm import tqdm


def one_hot_encode_it(df, nan_as_category = False, exclude_columns=[]):
    encode_these_coumns = []
    for col in df:
        if col == 'TARGET':
            continue
        x = df[col].nunique()
        if (2 <= x <= 4):
            encode_these_coumns.append(col)
    encode_these_coumns = list(set(encode_these_coumns) - set(exclude_columns))
    df = pd.get_dummies(df, columns= encode_these_coumns, dummy_na= nan_as_category)
    return df


def label_encode_it(df):
    encode_these_columns = []
    for col in list(df):
        if col == 'TARGET':
            continue
        if str(df[col].dtype) in ['object', 'category']:
            encode_these_columns.append(col)
            df[col] = df[col].astype('category').cat.codes
    print(encode_these_columns, '**********')
    #label_encoder = LabelEncoder(cols=encode_these_columns)
    #l_encoded =  label_encoder.fit_transform(df)
    return df

def quantile_cut(ds, ncut = 20):
  return pd.qcut(ds, ncut, labels=range(1, ncut+1))


def read_csv_data(file_name, debug, server=False, num_rows=200):
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

def read_hdf_data(file_name, debug=True, num_rows=200):
    try:
        path = '/home/science/data/'
        store = pd.HDFStore(path + file_name + '.h5')
    except OSError as e:
        path = '/home/gublu/Desktop/THINKSTATS/Competition/hdf/'
        store = pd.HDFStore(path + file_name + '.h5')
    if debug:
      df = store['data'].iloc[:num_rows, :]
    else:
      df = store['data']
      #df = pd.read_hdf(path + file_name + '.h5', 'data')
    for col in list(df):
        if str(df[col].dtype) == 'category':
            df[col] = df[col].astype('object')
    return df


def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype
        if str(col_type) not in ['object', 'category']:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    #print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


def add_features_in_group(features, gr_, feature_name, aggs, prefix):
    for agg in aggs:
        if agg == 'sum':
            features['{}{}_sum'.format(prefix, feature_name)] = gr_[feature_name].sum()
        elif agg == 'mean':
            features['{}{}_mean'.format(prefix, feature_name)] = gr_[feature_name].mean()
        elif agg == 'max':
            features['{}{}_max'.format(prefix, feature_name)] = gr_[feature_name].max()
        elif agg == 'min':
            features['{}{}_min'.format(prefix, feature_name)] = gr_[feature_name].min()
        elif agg == 'std':
            features['{}{}_std'.format(prefix, feature_name)] = gr_[feature_name].std()
        elif agg == 'count':
            features['{}{}_count'.format(prefix, feature_name)] = gr_[feature_name].count()
        elif agg == 'skew':
            features['{}{}_skew'.format(prefix, feature_name)] = skew(gr_[feature_name])
        elif agg == 'kurt':
            features['{}{}_kurt'.format(prefix, feature_name)] = kurtosis(gr_[feature_name])
        elif agg == 'iqr':
            features['{}{}_iqr'.format(prefix, feature_name)] = iqr(gr_[feature_name])
        elif agg == 'median':
            features['{}{}_median'.format(prefix, feature_name)] = gr_[feature_name].median()

    return features


def add_trend_feature(features, gr, feature_name, prefix):
    y = gr[feature_name].values
    try:
        x = np.arange(0, len(y)).reshape(-1, 1)
        lr = LinearRegression()
        lr.fit(x, y)
        trend = lr.coef_[0]
    except:
        trend = np.nan
    features['{}{}'.format(prefix, feature_name)] = trend
    return features


def get_feature_names_by_period(features, period):
    return sorted([feat for feat in features.keys() if '_{}_'.format(period) in feat])

def chunk_groups(groupby_object, chunk_size):
    n_groups = groupby_object.ngroups
    group_chunk, index_chunk = [], []
    for i, (index, df) in enumerate(groupby_object):
        group_chunk.append(df)
        index_chunk.append(index)

        if (i + 1) % chunk_size == 0 or i + 1 == n_groups:
            group_chunk_, index_chunk_ = group_chunk.copy(), index_chunk.copy()
            group_chunk, index_chunk = [], []
            yield index_chunk_, group_chunk_

def safe_div(a, b):
    try:
        return float(a) / float(b)
    except:
        return 0.0


def parallel_apply(groups, func, index_name='Index', num_workers=1, chunk_size=100000):
    n_chunks = np.ceil(1.0 * groups.ngroups / chunk_size)
    indeces, features = [], []
    for index_chunk, groups_chunk in tqdm(chunk_groups(groups, chunk_size), total=n_chunks):
        with mp.pool.Pool(num_workers) as executor:
            features_chunk = executor.map(func, groups_chunk)
        features.extend(features_chunk)
        indeces.extend(index_chunk)

    features = pd.DataFrame(features)
    features.index = indeces
    features.index.name = index_name
    return features


def pe(*z):
    print(*z)
    sys.exit(1)

def ht(df, num=1):
  pd.set_option('display.max_rows', None)
  print(df.head(num).T)
  sys.exit(1)

def dt(df):
  for col in df:
    print(col, ',', df[col].dtype, ',', df[col].min(),',', df[col].max(),',', df[col].nunique())
  sys.exit(1)

def p(*x):
  print(x)

def l(df):
  pe(list(df))

def pf(file_name):
    import sys
    sys.stdout = open(file_name, "w")

def summary(df):
  print('max', 'min', 'nunique', 'count', 'isnull_sum')
  for col in df:
    print(col, df[col].max(), df[col].min(), df[col].nunique(), df[col].count(), df[col].isnull().sum())
  sys.exit(1)

def write_to_file(params, auc = 999):
    with open('results/params/params_' + str(round(auc, 5)) + '.txt', "w") as text_file:
        for k, v in params.items():
            text_file.write(str(k) + ':'+ str(v) + ',\n')
