import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging 
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_filepath = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        try:
            numerical_columns = [
                'LIMIT_BAL', 'AGE', 'BILL_AMT_SEPT', 'BILL_AMT_AUG', 'BILL_AMT_JUL',
                'BILL_AMT_JUN', 'BILL_AMT_MAY', 'BILL_AMT_APR', 'PAY_AMT_SEPT',
                'PAY_AMT_AUG', 'PAY_AMT_JUL', 'PAY_AMT_JUN', 'PAY_AMT_MAY', 'PAY_AMT_APR', 'PAY_SEPT', 'PAY_AUG', 'PAY_JUL',
                'PAY_JUN', 'PAY_MAY', 'PAY_APR'
            ]
            categorical_columns = [
                'SEX', 'EDUCATION', 'MARRIAGE'
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info("Numerical columns standard scaling completed")
            logging.info("Categorical columns encoding completed")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Load training and testing data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Loaded train and test data for transformation")
    
            logging.info("Obtaining preprocessing object")
            preprocessor_obj = self.get_data_transformer_obj()
    
            # Prepare input and target features
            target_column_name = 'Is_Defaulter'
            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]
    
            logging.info("Applying preprocessing object on training and testing data")
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
    
            logging.info("Applying SMOTE to balance the training dataset")
            smote = SMOTE()
            input_feature_train_smote, target_feature_train_smote = smote.fit_resample(
                input_feature_train_arr, target_feature_train_df
            )
    
            logging.info(f"Original unbalanced training dataset shape: {len(target_feature_train_df)}")
            logging.info(f"Balanced training dataset shape after SMOTE: {len(target_feature_train_smote)}")
    
            logging.info("Splitting the balanced data into training and validation sets")
            input_feature_train_final, input_feature_valid, target_feature_train_final, target_feature_valid = train_test_split(
                input_feature_train_smote, target_feature_train_smote, test_size=0.2, random_state=42, stratify=target_feature_train_smote
            )
    
            logging.info(f"Saving preprocessing object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_filepath,
                obj=preprocessor_obj
            )
    
            return (
                input_feature_train_final, 
                input_feature_valid, 
                target_feature_train_final, 
                target_feature_valid, 
                input_feature_test_arr, 
                target_feature_test_df
            )
    
        except Exception as e:
            raise CustomException(e, sys)
    