#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 20:12:14 2022

@author: apple
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import keras
from sklearn.metrics import r2_score,mean_squared_error, mean_absolute_error
from math import sqrt


    
def PreData(self):
        #Rename Columns 
        self.rename(columns = {'Unnamed: 0':'Date'}, inplace = True)      
        self.rename(columns = {'demand [MW]':'Y'}, inplace = True)
                
        #Check duplicated
        self[self.duplicated(keep=False)]

        #Drop NA
        df = self.dropna(axis=0)

        #Format datetime
        df.Date=pd.to_datetime(df.Date,utc=True)

        # Sorting data in ascending order by the date
        df = df.sort_values(by='Date')

        #Set Date as index
        df.set_index('Date', inplace=True)
        print(df.info())
        return df

    


def ScaleData(self):
    df =self[["Y","albedo [%]"]]
    df = df.drop("albedo [%]",1)
    df_train, df_test= np.split(df, [int(.8 *len(df))])
    # Data preprocess
    training_set = df_train.values
    training_set = np.reshape(training_set, (len(training_set), 1))

    sc = MinMaxScaler()
    training_set = sc.fit_transform(training_set)
    X_train = training_set[0:len(training_set)-1]
    y_train = training_set[1:len(training_set)]
    X_train = np.reshape(X_train, (len(X_train), 1, 1))


    return [df, X_train, y_train, df_test, df_train, sc]



def FitLSTM(X_train,y_train):
    
    early_stopping=keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                              patience=2, verbose=0, mode='auto',
                              baseline=None, restore_best_weights=False)
    # Start neural network
    model = Sequential()

    model.add(LSTM(128,activation="relu",input_shape=(1,1))) 
    model.add(Dropout(0.2)) #delete hidden layers randomly
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_split=0.33, epochs=50, batch_size=512,verbose=2,callbacks = [early_stopping]) 
    #history = model.fit(X_train, y_train, validation_split=0.33, epochs=5, batch_size=512,verbose=2,callbacks = [early_stopping]) 
    return [history,model]
        


def LossPlot(self):
    # summarize history for loss
    loss = self.history['loss']
    valloss = self.history['val_loss']
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, valloss, "r", label="Validation loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    


def PredictDemand(self,sc,model):
    # Making the predictions
    test_set = self.values
    inputs = sc.transform(np.reshape(test_set, (len(test_set), 1)))
    inputs = np.reshape(inputs, (len(inputs), 1, 1))
    predicteddemand = model.predict(inputs)
    predicteddemand = sc.inverse_transform(predicteddemand)
    self['demand_Prediction'] = predicteddemand
    return self


def PredictNew(self,sc,model):
    # Making the predictions
    test_set = self.values
    
    inputs = sc.transform(np.reshape(test_set, (len(test_set), 1)))
    inputs = np.reshape(inputs, (len(inputs), 1, 1))
    predicteddemand = model.predict(inputs)
    predicteddemand = sc.inverse_transform(predicteddemand)
    self['demand_Prediction'] = predicteddemand
    return self

def Score(df_test):
    MSE = mean_squared_error(y_true=df_test['demand_Prediction'],y_pred=df_test['Y'])
        
    print("MSE: ",MSE)
    MAE = mean_absolute_error(y_true=df_test['demand_Prediction'],y_pred=df_test['Y'])   
    print("MAE: ",MAE)
        
    RMSE = sqrt(MSE)
    print('RMSE : %f' % RMSE)
     
    R2_SCORE=r2_score(df_test['demand_Prediction'], y_pred=df_test['Y']) 
    print('R2_SCORE : %f' % R2_SCORE)
    return[MSE, MAE, RMSE, R2_SCORE]




