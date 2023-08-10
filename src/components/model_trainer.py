import os 
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.neighbors import NearestCentroid
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
import pickle
from src.utils import save_object,evaluate_models
from sklearn.metrics import f1_score,accuracy_score

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Spliting training and test input data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestClassifier(),
            }
            params={
                "Random Forest": {
                            'n_estimators': [50, 100, 200,300],
                            'max_depth': [5, 10, 20, 15],
                            'min_samples_split': [2, 5],
                            'min_samples_leaf': [1, 2]
                }}
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")
            save_object(
                            file_path=self.model_trainer_config.trained_model_file_path,
                            obj=best_model
                        )
            predicted=best_model.predict(X_test)
            accuracy_test = accuracy_score(y_test,predicted)
            return accuracy_test


        except Exception as e:
            raise CustomException(e,sys)