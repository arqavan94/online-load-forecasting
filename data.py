from numpy import isnan
import numpy as np
import pandas as pd
from numpy import nan
from sklearn.preprocessing import MinMaxScaler

split_dataset_value=0.65

def load_data(url):
 
  df = pd.read_csv(url,sep=',')
  df['date_time']= pd.to_datetime(df['date_time'])
  df=df.set_index('date_time')
  if(df.isnull().values.any()):
    df.replace('NaN', nan, inplace=True)
  df = df.astype('float32')
  return df

def fill_missing(values):
  one_day=24
  for row in range(values.shape[0]):
    for col in range(values.shape[1]):
      if isnan(values[row,col]):
        values[row,col]= values[row- one_day,col]

def to_daily_data(df):
  daily_groups = df.resample('D')
  daily_data = daily_groups.sum()
  return daily_data

def split_test_train(df,p):
  train_size = int(len(df) *p)
  train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
  return train,test

def create_dataset(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps),0:].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps:i + time_steps+1])
    return np.array(Xs), np.array(ys)

df=load_data('F:\online_project\clean_data_32100497.csv')
fill_missing(df.values)
dataset=to_daily_data(df)
# online_size= int(0.2 * len(dataset))
# online_data, dataset= dataset.iloc[(len(dataset)-online_size):], dataset.iloc[0:len(dataset)-online_size]
train,test=split_test_train(dataset,split_dataset_value)

def transformer_X(train,test,f_columns):
  f_transformer = MinMaxScaler()
  f_transformer = f_transformer.fit(train[f_columns].to_numpy())
  train.loc[:, f_columns] = f_transformer.transform(train[f_columns].to_numpy())
  test.loc[:, f_columns] = f_transformer.transform(test[f_columns].to_numpy())
  
  cnt_transformer = MinMaxScaler()
  cnt_transformer = cnt_transformer.fit(train[['Active_Power']])
  train['Active_Power'] = cnt_transformer.transform(train[['Active_Power']])
  test['Active_Power'] = cnt_transformer.transform(test[['Active_Power']])
  return train,test,cnt_transformer

f_columns = ['temp']
train,test,cnt_transformer= transformer_X(train,test,f_columns)