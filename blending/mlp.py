### FINALIZED
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier


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


seed = 7
np.random.seed(seed)


df = read_csv_data('shiv', debug=False)
df = df.replace(-np.inf, np.nan)
df = df.replace(np.inf, np.nan)

cols = [col for col in df.columns if col not in ['TARGET', 'SK_ID_CURR']]
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
print(cols_to_drop, 'cols_to_drop')
print(train.shape, test.shape)
train_dataset = train.values
X = train_dataset[:,2:]
y = train_dataset[:,1]
y=y.astype('int')
print(X)

test_dataset = test.values
X_test = test_dataset[:,2:]
print(type(X_test))


print(X.shape, y.shape, X_test.shape)


# In[34]:
# https://www.kaggle.com/aharless/simple-ffnn-from-dromosys-features

print( 'Setting up a multilayer perceptron...' )

mlp = MLPClassifier(activation='relu', alpha=80, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(500, 150, 30, 2), learning_rate='constant',
       learning_rate_init=0.001, max_iter=500, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='lbfgs', tol=0.0001, validation_fraction=0.2, verbose=True,
       warm_start=False)


print( 'Fitting neural network...' )
mlp.fit(X, y, validation_split=0.2, epochs=100, batch_size=10, verbose=2)

print( 'Predicting...' )
y_pred_train = mlp.predict(X).flatten().clip(0,1)
y_pred_test = mlp.predict(X_test).flatten().clip(0,1)

tr = pd.DataFrame()
tr['SK_ID_CURR'] = train['SK_ID_CURR']
tr['MLP_SCORE'] = y_pred_train
#tr[['SK_ID_CURR', 'NN_SCORE']].to_csv('sub_nn.csv', index= False)

print( 'Saving results...' )
te = pd.DataFrame()
te['SK_ID_CURR'] = test['SK_ID_CURR']
te['MLP_SCORE'] = y_pred_test
te[['SK_ID_CURR', 'TARGET']].to_csv('sub_mlp.csv', index= False)

tr_te = tr.append(te).reset_index()
tr_te.to_csv('mlp_tr_te.csv', index= False)

print( tr_te.head() )
