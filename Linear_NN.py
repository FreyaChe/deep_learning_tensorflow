#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 16:57:42 2025

@author: freya
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
import matplotlib.pyplot as plt

# In[preprocess data]
# load data 
dataset = pd.read_csv("/Users/freya/Study/self-learning/Kaggle_house-prices/train.csv")
dataset_df = dataset.drop('Id', axis=1)

# exclude nan value
missing = dataset_df.isna().sum().sort_values(ascending=False)
cols_drop_missing = missing[missing > dataset_df.shape[0]*0.2].index # exclude variance which nan over 20%
dataset_df = dataset_df.drop(columns=cols_drop_missing)

# convert str variance to num
df_categorical = dataset_df.select_dtypes(exclude=[np.number])
label_encoders = {}
for col in df_categorical.columns:
    dataset_df[col] = dataset_df[col].fillna("Missing")
    le = LabelEncoder()
    dataset_df[col] = le.fit_transform(dataset_df[col].astype(str))
    label_encoders[col] = le
    
# correlation between variance and price
corr_with_price = abs(dataset_df.corr()['SalePrice']).sort_values(ascending=False)
cols_drop_lowcorr = corr_with_price[corr_with_price < 0.05].index # exclude corr lower than 0.05

# exclude data
cols_to_drop = list(set(cols_drop_missing) | set(cols_drop_lowcorr)) # for later test session
dataset_df = dataset_df.drop(columns=cols_drop_lowcorr)

# check data
# numeric_cols = dataset_df.columns.tolist()
# numeric_cols.remove("SalePrice")   # avoid SalePrice vs SalePrice

# for feature in numeric_cols:
#     plt.figure(figsize=(6,4))
#     plt.scatter(dataset_df[feature], dataset_df["SalePrice"], s=10, alpha=0.5)
#     plt.title(f"{feature} vs SalePrice", fontsize=14)
#     plt.xlabel(feature, fontsize=12)
#     plt.ylabel("SalePrice", fontsize=12)
#     plt.grid(alpha=0.2)
#     plt.show()

dataset_df.loc[dataset_df['LotFrontage'] > 300, 'LotFrontage'] = np.nan
dataset_df.loc[dataset_df['LotArea'] > 100000, 'LotArea'] = np.nan
dataset_df.loc[dataset_df['MasVnrArea'] > 1600, 'MasVnrArea'] = np.nan
dataset_df.loc[dataset_df['BsmtFinSF1'] > 5000, 'BsmtFinSF1'] = np.nan
dataset_df.loc[dataset_df['TotalBsmtSF'] > 6000, 'TotalBsmtSF'] = np.nan
dataset_df.loc[dataset_df['1stFlrSF'] > 4000, '1stFlrSF'] = np.nan
dataset_df.loc[dataset_df['GrLivArea'] > 4500,'GrLivArea'] = np.nan
dataset_df.loc[dataset_df['BedroomAbvGr'] > 7, 'BedroomAbvGr'] = np.nan

# replace nan as mean
df_numeric = dataset_df.select_dtypes(include=[np.number])
dataset_df[df_numeric.columns] = df_numeric.fillna(df_numeric.mean())

# standard data
x = dataset_df.drop('SalePrice', axis=1)
x = StandardScaler().fit_transform(x)
y_log = np.log1p(dataset_df['SalePrice']) # convert sale price, back：y_real = np.expm1(y_log)

del dataset, dataset_df, df_numeric, df_categorical


# In[model & training]
num_input = x.shape[1]
num_outputs = 1
num_hidden1 = 40
num_hidden2 = 24
num_hidden3 = 5
dropout1, dropout2 = 0.2, 0.4

initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(), # input layer
    tf.keras.layers.Dense(
        num_hidden1, activation='relu', 
        kernel_initializer=initializer
        ), # hiden layer, with initial weight
    tf.keras.layers.Dropout(dropout1),
    tf.keras.layers.Dense(num_hidden2, 
                          activation='relu', 
                          kernel_initializer=initializer), # hiden layer
    tf.keras.layers.Dropout(dropout2),
    tf.keras.layers.Dense(num_hidden3, 
                          activation='relu', 
                          kernel_initializer=initializer), # hiden layer
    tf.keras.layers.Dense(num_outputs)]) # output layer

model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
)

callback = tf.keras.callbacks.EarlyStopping(patience=50, mode='min', 
                                            restore_best_weights=True, start_from_epoch=150)

history = model.fit(x, y_log, batch_size=10, epochs=1000, validation_split=0.1, callbacks=[callback])


# In[test session]
testset = pd.read_csv("/Users/freya/Study/self-learning/Kaggle_house-prices/test.csv")
dataset_df = testset.drop('Id', axis=1)
dataset_df = dataset_df.drop(columns=cols_to_drop)
test_ID = testset['Id']

df_numeric = dataset_df.select_dtypes(include=[np.number])
dataset_df[df_numeric.columns] = df_numeric.fillna(df_numeric.mean())

df_categorical = dataset_df.select_dtypes(exclude=[np.number])
label_encoders = {}
for col in df_categorical.columns:
    dataset_df[col] = dataset_df[col].fillna("Missing")
    le = LabelEncoder()
    dataset_df[col] = le.fit_transform(dataset_df[col].astype(str))
    label_encoders[col] = le
    
scaler = StandardScaler()
test_data = scaler.fit_transform(dataset_df)

y_pred = model.predict(test_data)
y_real = np.expm1(y_pred)

sample_submission_df = pd.read_csv('/Users/freya/Study/self-learning/Kaggle_house-prices/sample_submission.csv')
sample_submission_df['SalePrice'] = y_real
sample_submission_df.to_csv('/Users/freya/Study/self-learning/Kaggle_house-prices/submission.csv', index=False)
