#Model Logistic Regression(scikit). Dataset: Log(X+1)


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

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


def log_x_plus_one(df):
    cols = [col for col in df.columns if col not in ['TARGET', 'SK_ID_CURR']]
    for col in cols:
        try:
            df[col]  = np.log(df[col] + 1)
        except:
            df = df.drop(col, 1)
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
df[cols] = label_encode_it(df[cols])
df = log_x_plus_one(df)

train = df[df['TARGET'].notnull()]
test = df[df['TARGET'].isnull()]

train  = train.fillna(df.mean())
test  = test.fillna(df.mean())

cols_to_drop = []
for col in list(df):
    if col in ['TARGET', 'SK_ID_CURR']:
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
X_test = test_dataset[:,2:]
print(type(X_test))
print('X.shape, y.shape, X_test.shape', X.shape, y.shape, X_test.shape)




# In[5]:
df = pd.DataFrame({"SK_ID_CURR": df['SK_ID_CURR']})

print('Linear Regression ****************')
lr = LogisticRegression()
lr_train = lr.fit(X, y)
lr_X_prediction  = lr.predict_proba(X)
lr_X_test_prediction  = lr.predict_proba(X_test)
tr_te_concatenated = numpy.concatenate([lr_X_prediction, lr_X_test_prediction])
df['lr'] = preds

print('final tr_te shape', df.shape)
print(df.head())

df.to_csv('lr_tr_te.csv', index= False)


