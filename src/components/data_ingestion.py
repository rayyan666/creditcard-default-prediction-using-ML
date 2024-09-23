import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_trasformation import DataTransformation
from src.components.data_trasformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('data/data/UCI_Credit_Card.csv')
            logging.info('Read the dataset as dataframe')

            # Renaming columns
            df.rename(columns={'default.payment.next.month': 'Is_Defaulter'}, inplace=True)
            df.rename(columns={
                'PAY_0': 'PAY_SEPT', 'PAY_2': 'PAY_AUG', 'PAY_3': 'PAY_JUL', 'PAY_4': 'PAY_JUN', 
                'PAY_5': 'PAY_MAY', 'PAY_6': 'PAY_APR',
                'BILL_AMT1': 'BILL_AMT_SEPT', 'BILL_AMT2': 'BILL_AMT_AUG', 'BILL_AMT3': 'BILL_AMT_JUL', 
                'BILL_AMT4': 'BILL_AMT_JUN', 'BILL_AMT5': 'BILL_AMT_MAY', 'BILL_AMT6': 'BILL_AMT_APR',
                'PAY_AMT1': 'PAY_AMT_SEPT', 'PAY_AMT2': 'PAY_AMT_AUG', 'PAY_AMT3': 'PAY_AMT_JUL',
                'PAY_AMT4': 'PAY_AMT_JUN', 'PAY_AMT5': 'PAY_AMT_MAY', 'PAY_AMT6': 'PAY_AMT_APR'
            }, inplace=True)

            # Replacing categorical values
            df.replace({'SEX': {1: 0, 2: 1}}, inplace=True)
            df.replace({'EDUCATION': {1: 'Graduate School', 2: 'University', 3: 'High School', 4: 'Others'}}, inplace=True)
            df.replace({'MARRIAGE': {1: 'Married', 2: 'Single', 3: 'Others'}}, inplace=True)

            # Adjusting EDUCATION category
            fil = (df['EDUCATION'] == 0) | (df['EDUCATION'] == 5) | (df['EDUCATION'] == 6) | (df['EDUCATION'] == 4)
            df.loc[fil, 'EDUCATION'] = 'Others'

            # Adjusting MARRIAGE category
            fil = df['MARRIAGE'] == 0
            df.loc[fil, 'MARRIAGE'] = 'Others'

            #Dropping the I.D column
            df.drop(columns=['ID'], inplace=True)

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Is_Defaulter'])

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    # Initiating data ingestion
    data_ingestion = DataIngestion()
    train_data, test_data = data_ingestion.initiate_data_ingestion()

    # Initiating data transformation
    data_transformation = DataTransformation()
    X_train, X_valid, y_train, y_valid, X_test, y_test = data_transformation.initiate_data_transformation(
        train_data, test_data
    )

    # Initiating model training
    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(X_train, X_valid, y_train, y_valid))