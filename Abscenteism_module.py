import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

class CustomScaler(BaseEstimator, TransformerMixin): 
    
    def __init__(self, columns_to_scale):
        self.scaler = StandardScaler()
        self.columns_to_scale = columns_to_scale
        
    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns_to_scale], y)
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
        self.df_with_predictions = df.copy()
        df = df.drop(['ID'], axis = 1)
        df['Absenteeism Time in Hours'] = 'NaN'

        reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first = True)
        reason_type_1 = reason_columns.loc[:,1:14].max(axis=1)
        reason_type_2 = reason_columns.loc[:,15:17].max(axis=1)
        reason_type_3 = reason_columns.loc[:,18:21].max(axis=1)
        reason_type_4 = reason_columns.loc[:,22:].max(axis=1)
        df = df.drop(['Reason for Absence'], axis = 1)
        df = pd.concat([df, reason_type_1, reason_type_2, reason_type_3, reason_type_4], axis = 1)
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

        # re-order the columns in df
        column_names_upd = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Month Value', 'Day of the Week',
                                'Transportation Expense', 'Distance to Work', 'Age',
                                'Daily Work Load Average', 'Body Mass Index', 'Education', 'Children',
                                'Pets', 'Absenteeism Time in Hours']
        df = df[column_names_upd]


        df['Education'] = df['Education'].map({1:0, 2:1, 3:1, 4:1})
        df = df.fillna(value=0)
        # drop the original absenteeism time
        df = df.drop(['Absenteeism Time in Hours'],axis=1)
            
            # drop the variables we decide we don't need
        #df = df.drop(['Day of the Week','Daily Work Load Average','Distance to Work'],axis=1)
            
            # we have included this line of code if you want to call the 'preprocessed data'
        self.preprocessed_data = df.copy()
            
            # we need this line so we can use it in the next functions
        self.data = self.scaler.transform(df)
    
        # a function which outputs the probability of a data point to be 1
    def predicted_probability(self):
        if (self.data is not None):  
            pred = self.reg.predict_proba(self.data)[:,1]
            return pred
        
        # a function which outputs 0 or 1 based on our model
    def predicted_output_category(self):
        if (self.data is not None):
            pred_outputs = self.reg.predict(self.data)
            return pred_outputs
        
        # predict the outputs and the probabilities and 
        # add columns with these values at the end of the new data
        def predicted_outputs(self):
            if (self.data is not None):
                self.preprocessed_data['Probability'] = self.reg.predict_proba(self.data)[:,1]
                self.preprocessed_data ['Prediction'] = self.reg.predict(self.data)
                return self.preprocessed_data
