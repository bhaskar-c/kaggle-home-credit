import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import gc

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
def label_encode_it(df):
    encode_these_columns = []
    for col in list(df):
        if col == 'TARGET':
            continue
        if str(df[col].dtype) in ['object', 'category']:
            encode_these_columns.append(col)
            df[col] = df[col].astype('category').cat.codes
    print(encode_these_columns, '**********')
    return df
def min_max_scale_it(df):
    cols = [col for col in df.columns if col not in ['TARGET', 'SK_ID_CURR']]
    for col in cols:
        try:
            df[col]  = df[col].fillna(df[col].mean())
            df[col]=(df[col]-df[col].min())/(df[col].max()-df[col].min())
        except:
            pass
    return df


def norm_scale_it(df):
    cols = [col for col in df.columns if col not in ['TARGET', 'SK_ID_CURR']]
    for col in cols:
        try:
            df[col]  = df[col].fillna(df[col].mean())
            df[col] = (df[col] - df[col].mean()) / (df[col].max() - df[col].min())
        except:
            pass
    return df

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


# In[2]:


df = read_csv_data('shiv', debug=False)


# In[ ]:


# impute and scale
df = df.replace(-np.inf, np.nan)
df = df.replace(np.inf, np.nan)

cols = [col for col in df.columns if col not in ['TARGET', 'SK_ID_CURR']]
df  = norm_scale_it(df)
df  = min_max_scale_it(df)
df[cols] = label_encode_it(df[cols])

train = df[df['TARGET'].notnull()]
test = df[df['TARGET'].isnull()]

train  = train.fillna(df.mean())
test  = test.fillna(df.mean())



cols_to_drop = []
for col in list(df):
    if col == 'TARGET':
        continue
    train[col].replace(np.inf, np.nan, inplace=True)
    test[col].replace(np.inf, np.nan, inplace=True)
    train[col].replace(-np.inf, np.nan, inplace=True)
    test[col].replace(-np.inf, np.nan, inplace=True)
    if train[col].isnull().any():
        cols_to_drop.append(col)
    if test[col].isnull().any():
        cols_to_drop.append(col)
    cols_to_drop = list(set(cols_to_drop))

train.drop(cols_to_drop, axis=1, inplace=True)
test.drop(cols_to_drop, axis=1, inplace=True)
test = test.reset_index(drop=True)
print(test.head())
print(test.shape)
#print(cols_to_drop, 'cols_to_drop')
print(train.shape, test.shape)
train_dataset = train.values
X = train_dataset[:,2:]
y = train_dataset[:,1]
y=y.astype('int')
#print(X)


# In[4]:


train_dataset = train.values
X = train_dataset[:,2:]
y = train_dataset[:,1]
y=y.astype('int')
test_dataset = test.values
X_te = test_dataset[:,2:]
print(type(X_te))
print('X.shape, y.shape, X_te.shape', X.shape, y.shape, X_te.shape)




'''
It is highly recommended to use another dimensionality reduction
 |  method (e.g. PCA for dense data or TruncatedSVD for sparse data)
 |  to reduce the number of dimensions to a reasonable amount (e.g. 50)
'''
n_components = 20
pca = PCA(n_components=n_components)
principalComponents = pca.fit_transform(X)
col_names  = ["pc" + str(col+1) for col in range(n_components)]
tr_pca  = pd.DataFrame(data = principalComponents, columns = col_names)
te_pca_np = pca.transform(X_te)
te_pca  = pd.DataFrame(data = te_pca_np, columns = col_names)
print('tr_pca shape*****', tr_pca.shape)
print(tr_pca.head())

print('te_pca shape*****', te_pca.shape)
print(te_pca.head())

X_train = tr_pca.values
X_test = te_pca.values
output_df = pd.DataFrame({"SK_ID_CURR": df['SK_ID_CURR']})


del (df, train, test, X, X_te)
gc.collect()


kernels = ['linear', 'poly']

for kernel in kernels:
    print('Calculating for {} kernel****************', kernel)
    svc = SVC(kernel=kernel, probability=True)
    svc_train = svc.fit(X_train, y)
    svc_X_train_prediction  = svc.predict_proba(X_train)[:, 1]
    svc_X_test_prediction  = svc.predict_proba(X_test)[:, 1]
    tr_te_concatenated = np.concatenate([svc_X_train_prediction,svc_X_test_prediction])
    output_df['pca20_svm_'+ kernel + '_kernel' ] = tr_te_concatenated

print('final tr_te shape', output_df.shape)
output_df.to_csv('pca20_svm_tr_te.csv', index= False)
print(output_df.head())




