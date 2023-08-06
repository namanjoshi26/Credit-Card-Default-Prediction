import sys
import logging
from src.logger import logging
def error_message_detail(error, error_detail:sys):
    #error_detail will be present inside the sys
    _,_,exc_tb = error_detail.exc_info()
    # exc_tb will have all the info regarding the exception
    file_name = exc_tb.tb_frame.f_code.co_filename
    # here inside exc_tb exists tb_frame then -> f_code -> co_filename
    error_message="Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name,exc_tb.tb_lineno,str(error)
    )
    return error_message
    

class CustomException(Exception):
    def __init__(self, error_message,error_detail:sys): # default constructor
        super().__init__(error_message) # inherit the init function
        self.error_message= error_message_detail(error_message,error_detail=error_detail)

    def __str__(self):   #used for printing the error msg
        return self.error_message  
    
# if __name__=="__main__":  #testing exception
#     try:
#         a=1/0
#     except Exception as e:
#         logging.info("Divide by zero error")
#         raise CustomException(e,sys)
    #logging.info("Logging has started")