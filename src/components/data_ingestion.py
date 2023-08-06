import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from dataclasses import dataclass

@dataclass   #decorator
#generally we define variables in class by__init__, but with
#decorator we can define variables directly
class DataIngestionConfig:
    #Any i/p i.e. required we will give through this class
    train_data_path: str=os.path.join('artifacts',"train.csv")
    #artifact folder is created so that we can see the output(here all outputs are stored)
    test_data_path: str=os.path.join('artifacts',"test.csv")
    #raw_data_path: str=os.path.join('artifacts',"raw.csv") if we have raw file
    #above are the inputs that we are giving to data ingestion component
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig() #all the path will be save in this var

    def initiate_data_ingestion(self):
        #this function is for reading the code from database
        logging.info("Entered the data ingestion method or component")
        try:
    
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            #df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True) if we have raw file
            logging.info("Train and test data")
            test_set = pd.read_csv('Notebooks\data\credit_card_dataset_test.csv') #it can be from api,mongo,DB etc
            train_set = pd.read_csv('Notebooks\data\credit_card_dataset_train.csv')
        
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            
            logging.info("ingestion of data is complete")
            return(self.ingestion_config.train_data_path,
                   self.ingestion_config.test_data_path) #for data transformation
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()

