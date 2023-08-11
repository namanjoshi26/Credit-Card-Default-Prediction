from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application = Flask(__name__) #will give the entrypoint where we will execute it
app = application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') #it will search for template folder

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(
        LIMIT_BAL = float(request.form.get("LIMIT_BAL"))  ,
        SEX=int(request.form.get("SEX")) ,
        EDUCATION = int(request.form.get("EDUCATION"))  ,
        MARRIAGE = int(request.form.get("MARRIAGE")) , 
        AGE = float(request.form.get("AGE"))  ,
        PAY_0 = float(request.form.get("PAY_0") ),
        PAY_2 = float(request.form.get("PAY_2") ) ,
        PAY_3 = float(request.form.get("PAY_3"))  ,
        PAY_4 = float(request.form.get("PAY_4"))  ,
        PAY_5 = float(request.form.get("PAY_5") ) ,
        PAY_6 = float(request.form.get("PAY_6") ) ,
        BILL_AMT1 = float(request.form.get("BILL_AMT1"))  ,
        BILL_AMT2 = float(request.form.get("BILL_AMT2") ) ,
        BILL_AMT3 = float(request.form.get("BILL_AMT3")) ,
        PAY_AMT1 = float(request.form.get("PAY_AMT1") ) ,
        PAY_AMT2 = float(request.form.get("PAY_AMT2")) ,
        PAY_AMT3 = float(request.form.get("PAY_AMT3"))
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html',results=results[0])
    
#to test
if __name__ =="__main__":
    app.run(host="0.0.0.0",debug=True) #it will map it to 127....