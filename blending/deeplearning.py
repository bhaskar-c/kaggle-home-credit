
# coding: utf-8

# In[52]:


import numpy as np
import tensorflow as tf
from keras import backend as K
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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

seed = 7
np.random.seed(seed)


df = read_csv_data('shiv', debug=False)
df = df.replace(-np.inf, np.nan)
df = df.replace(np.inf, np.nan)

df = label_encode_it(df)
for col in list(df):
    if col == 'TARGET':
        continue
    df[col].replace(np.inf, np.nan, inplace=True)
    df[col].replace(-np.inf, np.nan, inplace=True)
    if df[col].isnull().any():
        df.drop(col, axis=1, inplace=True)

train = df[df['TARGET'].notnull()]
test = df[df['TARGET'].isnull()]
train  = train.fillna(df.mean())
test  = test.fillna(df.mean())

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


# baseline model
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(30, input_dim=X.shape[1], kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy']) # , auc])
    return model


# In[35]:


estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=1, batch_size=100, verbose=1)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, y, cv=kfold)
print("Smaller: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# In[ ]:


#test_df = read_csv_data('shiv_test', debug=True)


# In[43]:

estimators[1][1].fit(X, y)

y_pred = estimators[1][1].predict(X_test)#[:, 1]
#y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
output = pd.DataFrame(data={"SK_ID_CURR":test["SK_ID_CURR"],"TARGET":y_pred})
output.to_csv("submission.csv", index=False)
