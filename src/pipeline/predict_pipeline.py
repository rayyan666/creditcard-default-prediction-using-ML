import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=("artifacts\model.pkl")
            preprocessor_path=('artifacts\preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 LIMIT_BAL: float,
                 SEX: int,
                 AGE: int,
                 BILL_AMT_SEPT: float,
                 BILL_AMT_AUG: float,
                 BILL_AMT_JUL: float,
                 BILL_AMT_JUN: float,
                 BILL_AMT_MAY: float,
                 BILL_AMT_APR: float,
                 PAY_AMT_SEPT: float,
                 PAY_AMT_AUG: float,
                 PAY_AMT_JUL: float,
                 PAY_AMT_JUN: float,
                 PAY_AMT_MAY: float,
                 PAY_AMT_APR: float,
                 EDUCATION: str,
                 MARRIAGE: str,
                 PAY_SEPT: str,
                 PAY_AUG: str,
                 PAY_JUL: str,
                 PAY_JUN: str,
                 PAY_MAY: str,
                 PAY_APR: str):
        self.LIMIT_BAL = LIMIT_BAL
        self.SEX = SEX
        self.AGE = AGE
        self.BILL_AMT_SEPT = BILL_AMT_SEPT
        self.BILL_AMT_AUG = BILL_AMT_AUG
        self.BILL_AMT_JUL = BILL_AMT_JUL
        self.BILL_AMT_JUN = BILL_AMT_JUN
        self.BILL_AMT_MAY = BILL_AMT_MAY
        self.BILL_AMT_APR = BILL_AMT_APR
        self.PAY_AMT_SEPT = PAY_AMT_SEPT
        self.PAY_AMT_AUG = PAY_AMT_AUG
        self.PAY_AMT_JUL = PAY_AMT_JUL
        self.PAY_AMT_JUN = PAY_AMT_JUN
        self.PAY_AMT_MAY = PAY_AMT_MAY
        self.PAY_AMT_APR = PAY_AMT_APR
        self.EDUCATION = EDUCATION
        self.MARRIAGE = MARRIAGE
        self.PAY_SEPT = PAY_SEPT
        self.PAY_AUG = PAY_AUG
        self.PAY_JUL = PAY_JUL
        self.PAY_JUN = PAY_JUN
        self.PAY_MAY = PAY_MAY
        self.PAY_APR = PAY_APR

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "LIMIT_BAL": [self.LIMIT_BAL],
                "SEX": [self.SEX],
                "AGE": [self.AGE],
                "BILL_AMT_SEPT": [self.BILL_AMT_SEPT],
                "BILL_AMT_AUG": [self.BILL_AMT_AUG],
                "BILL_AMT_JUL": [self.BILL_AMT_JUL],
                "BILL_AMT_JUN": [self.BILL_AMT_JUN],
                "BILL_AMT_MAY": [self.BILL_AMT_MAY],
                "BILL_AMT_APR": [self.BILL_AMT_APR],
                "PAY_AMT_SEPT": [self.PAY_AMT_SEPT],
                "PAY_AMT_AUG": [self.PAY_AMT_AUG],
                "PAY_AMT_JUL": [self.PAY_AMT_JUL],
                "PAY_AMT_JUN": [self.PAY_AMT_JUN],
                "PAY_AMT_MAY": [self.PAY_AMT_MAY],
                "PAY_AMT_APR": [self.PAY_AMT_APR],
                "EDUCATION": [self.EDUCATION],
                "MARRIAGE": [self.MARRIAGE],
                "PAY_SEPT": [self.PAY_SEPT],
                "PAY_AUG": [self.PAY_AUG],
                "PAY_JUL": [self.PAY_JUL],
                "PAY_JUN": [self.PAY_JUN],
                "PAY_MAY": [self.PAY_MAY],
                "PAY_APR": [self.PAY_APR],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)