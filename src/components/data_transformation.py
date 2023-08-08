import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")

# from sklearn.base import BaseEstimator, TransformerMixin

# class OutlierCapper(BaseEstimator, TransformerMixin):
#     def __init__(self, columns):
#         self.columns = columns
    
#     def fit(self, X, y=None):
#         return self
    
#     def transform(self, X):
#         X_out = X.copy()
#         for column in self.columns:
#             q1, q3 = np.percentile(X_out[column], [25, 75])
#             iqr = q3 - q1
#             lower = (q1 - (1.5 * iqr))
#             upper = (q3 + (1.5 * iqr))
            
#             X_out[column] = np.where(X_out[column] < lower, lower, X_out[column])
#             X_out[column] = np.where(X_out[column] > upper, upper, X_out[column])
        
#         return X_out

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function is responsible for data transformation"""
        try:
            numerical_columns = []
            for i in range(1,7):
                feature1='BILL_AMT'+str(i)
                feature2='PAY_AMT'+str(i)
                numerical_columns.append(feature1)
                numerical_columns.append(feature2)
            numerical_columns.extend(['AGE','LIMIT_BAL'])

            num_pipeline= Pipeline(
                steps=[
                ("scaler",StandardScaler())

                ]
            )
            logging.info(f"Numerical columns: {numerical_columns}")
            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns)
                ]


            )
            return preprocessor
            
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()
            train_df['MARRIAGE']=train_df['MARRIAGE'].replace({0:3})
            test_df['MARRIAGE']=test_df['MARRIAGE'].replace({0:3})
            train_df['EDUCATION']=train_df['EDUCATION'].replace({0:4,5:4,6:4})
            test_df['EDUCATION']=test_df['EDUCATION'].replace({0:4,5:4,6:4})

            target_column_name="default payment next month"
            numerical_columns = []
            for i in range(1,7):
                feature1='BILL_AMT'+str(i)
                feature2='PAY_AMT'+str(i)
                numerical_columns.append(feature1)
                numerical_columns.append(feature2)
            numerical_columns.extend(['AGE','LIMIT_BAL'])

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )



        except Exception as e:
            raise CustomException(e,sys)
