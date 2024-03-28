import sys
import os

import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_models(X_train,y_train,X_test,y_test,models,param):
    
    try:
        report = {}
        logging.info("GridSearchCV Started-----------")
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train,y_train)
            
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)
            
            train_model_score = r2_score(y_train,y_train_pred)            
            test_model_score = r2_score(y_test,y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train,y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test,y_test_pred))

            report[list(models.keys())[i]] = test_model_score
            
            logging.info(f"Model Name : {model}\n Train Score(R2):{train_model_score} ,Test Score(R2):{test_model_score}\nTrain Error(RMSE):{train_rmse} ,Test Error(RMSE):{test_rmse}\n============")
        logging.info(f"Report:\n{report}")
        return report
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try :
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)