import os
import sys
from dataclasses import dataclass

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, X_valid, y_train, y_valid):
        try:
            logging.info("Training and testing data received")

            # Define the XGBoost model
            xgb = XGBClassifier()
            
            # Hyperparameter Grid
            param_dict = {'learning_rate': [0.15, 0.1, 0.05],
                          'n_estimators' : [200, 250],
                          'max_depth' : [15,20,25],
                          'min_child_weight' : [1,3],
                          'gamma': [0.3, 0.2, 0.1]}

            # Grid search
            xgb_grid = RandomizedSearchCV(estimator=xgb,
                                          param_distributions=param_dict,
                                          n_jobs=-1, n_iter=5, cv=3,
                                          verbose=2, scoring='roc_auc')
            # Fitting model
            xgb_grid.fit(X_train, y_train)

            # Get the best model and its parameters
            xgb_optimal_model = xgb_grid.best_estimator_
            best_params = xgb_grid.best_params_
            logging.info(f"Best XGBoost model parameters: {best_params}")

            # Save the optimal model
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=xgb_optimal_model)

            # Evaluate the model on validation data
            y_valid_pred_xgb_grid = xgb_optimal_model.predict(X_valid)
            y_train_pred_xgb_grid = xgb_optimal_model.predict(X_train)

            # Metrics for training data
            accuracy_train = round(accuracy_score(y_train, y_train_pred_xgb_grid), 3)
            precision_train = round(precision_score(y_train, y_train_pred_xgb_grid), 3)
            recall_train = round(recall_score(y_train, y_train_pred_xgb_grid), 3)
            f1_train = round(f1_score(y_train, y_train_pred_xgb_grid), 3)
            auc_train = round(roc_auc_score(y_train, y_train_pred_xgb_grid), 3)

            # Metrics for validation data
            accuracy_valid = round(accuracy_score(y_valid, y_valid_pred_xgb_grid), 3)
            precision_valid = round(precision_score(y_valid, y_valid_pred_xgb_grid), 3)
            recall_valid = round(recall_score(y_valid, y_valid_pred_xgb_grid), 3)
            f1_valid = round(f1_score(y_valid, y_valid_pred_xgb_grid), 3)
            auc_valid = round(roc_auc_score(y_valid, y_valid_pred_xgb_grid), 3)

            logging.info(f"Train Accuracy: {accuracy_train:.4f}")
            logging.info(f"Train Precision: {precision_train:.4f}")
            logging.info(f"Train Recall: {recall_train:.4f}")
            logging.info(f"Train F1 Score: {f1_train:.4f}")
            logging.info(f"Train AUC: {auc_train:.4f}")

            logging.info(f"Validation Accuracy: {accuracy_valid:.4f}")
            logging.info(f"Validation Precision: {precision_valid:.4f}")
            logging.info(f"Validation Recall: {recall_valid:.4f}")
            logging.info(f"Validation F1 Score: {f1_valid:.4f}")
            logging.info(f"Validation AUC: {auc_valid:.4f}")

            return accuracy_valid, precision_valid, recall_valid, f1_valid, auc_valid

        except Exception as e:
            raise CustomException(e, sys)