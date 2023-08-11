import os
import sys
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import pickle

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            gs = GridSearchCV(model,param_grid=para,cv=cv,scoring='f1',n_jobs=-1)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)
            accuracy = accuracy_score(y_train, y_train_pred)
            # compute f1-score
            train_model_score = f1_score(y_train, y_train_pred)

            test_model_score = f1_score(y_test, y_test_pred)
            accuracy_test = accuracy_score(y_train, y_train_pred)

            report[list(models.keys())[i]] = accuracy_test

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj: #read byte mode
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)