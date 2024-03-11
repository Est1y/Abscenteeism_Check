from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import pickle

class CustomScaler(BaseEstimator, TransformerMixin): 
    def __init__(self, columns_to_scale):
        self.scaler = StandardScaler()
        self.columns_to_scale = columns_to_scale

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns_to_scale])
        return self

    def transform(self, X, y=None):
        X_scaled = X.copy()
        X_scaled[self.columns_to_scale] = self.scaler.transform(X[self.columns_to_scale])
        return X_scaled

class absenteeism_model():
    def __init__(self, model_file, scaler_file):
        with open('model','rb') as model_file, open('Scaler', 'rb') as scaler_file:
            self.reg = pickle.load(model_file)
            self.scaler = pickle.load(scaler_file)
            self.data = None
    
    def load_and_clean_data(self, data_file):
        df = pd.read_csv(data_file,delimiter=',')
        print(df)
        self.df_with_predictions = df.copy()
        df = df.drop(['ID'], axis = 1)
        df['Absenteeism Time in Hours'] = 'NaN'

        reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first = True)
      
        Reason_1 = reason_columns.loc[:,1:14].max(axis=1)
        Reason_2 = reason_columns.loc[:,15:17].max(axis=1)
        Reason_3 = reason_columns.loc[:,18:21].max(axis=1)
        Reason_4 = reason_columns.loc[:,22:].max(axis=1)
        df = df.drop(['Reason for Absence'], axis = 1)
        df = pd.concat([df, Reason_1, Reason_2, Reason_3, Reason_4], axis = 1)
        column_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',
                       'Daily Work Load Average', 'Body Mass Index', 'Education', 'Children',
                       'Pets', 'Absenteeism Time in Hours', 'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4']
        df.columns = column_names
        column_names_reordered = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4','Date', 'Transportation Expense', 'Distance to Work', 'Age',
                       'Daily Work Load Average', 'Body Mass Index', 'Education', 'Children',
                       'Pets', 'Absenteeism Time in Hours']
        df = df[column_names_reordered]

        # Convert 'Date' column to datetime
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        
        list_months = []
        for i in range(df.shape[0]):
            list_months.append(df['Date'][i].month)

        df['Month Value'] = list_months
        # Calculate the day of the week
        df['Day of the Week'] = df['Date'].apply(lambda x: x.weekday())

        # Drop the 'Date' column
        df = df.drop(['Date'], axis = 1)
        df = df.drop(['Day of the Week', 'Daily Work Load Average', 'Distance to Work'], axis = 1)

        # re-order the columns in df
        column_names_upd = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Month Value', 
                                'Transportation Expense', 'Age',
                                'Body Mass Index', 'Education', 'Children',
                                'Pets', 'Absenteeism Time in Hours']
        df = df[column_names_upd]


        df['Education'] = df['Education'].map({1:0, 2:1, 3:1, 4:1})
        df = df.fillna(value=0)
        # drop the original absenteeism time
        df = df.drop(['Absenteeism Time in Hours'],axis=1)
        Unscaled_input = df.iloc[ :, : -1]
        Unscaled_input.columns.values
        columns_to_omit = ['Result_1', 'Result_2', 'Result_3', 'Result_4','Education']
        columns_to_scale = [x for x in Unscaled_input.columns.values if x not in columns_to_omit]

        self.pipe = Pipeline([
            ('scaler', CustomScaler(columns_to_scale)),
        ])
        self.data = self.pipe.fit_transform(df)
    
        return self.data.shape