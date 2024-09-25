from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Extract form data
        data = CustomData(
            LIMIT_BAL=float(request.form.get('LIMIT_BAL')),
            SEX=int(request.form.get('SEX')),
            AGE=int(request.form.get('AGE')),
            BILL_AMT_SEPT=float(request.form.get('BILL_AMT_SEPT')),
            BILL_AMT_AUG=float(request.form.get('BILL_AMT_AUG')),
            BILL_AMT_JUL=float(request.form.get('BILL_AMT_JUL')),
            BILL_AMT_JUN=float(request.form.get('BILL_AMT_JUN')),
            BILL_AMT_MAY=float(request.form.get('BILL_AMT_MAY')),
            BILL_AMT_APR=float(request.form.get('BILL_AMT_APR')),
            PAY_AMT_SEPT=float(request.form.get('PAY_AMT_SEPT')),
            PAY_AMT_AUG=float(request.form.get('PAY_AMT_AUG')),
            PAY_AMT_JUL=float(request.form.get('PAY_AMT_JUL')),
            PAY_AMT_JUN=float(request.form.get('PAY_AMT_JUN')),
            PAY_AMT_MAY=float(request.form.get('PAY_AMT_MAY')),
            PAY_AMT_APR=float(request.form.get('PAY_AMT_APR')),
            EDUCATION=request.form.get('EDUCATION'),
            MARRIAGE=request.form.get('MARRIAGE'),
            PAY_SEPT=int(request.form.get('PAY_SEPT')),
            PAY_AUG=int(request.form.get('PAY_AUG')),
            PAY_JUL=int(request.form.get('PAY_JUL')),
            PAY_JUN=int(request.form.get('PAY_JUN')),
            PAY_MAY=int(request.form.get('PAY_MAY')),
            PAY_APR=int(request.form.get('PAY_APR')))
        
        
        # Convert to DataFrame
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        # Initialize the prediction pipeline and make predictions
        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")

         # Determine if the result indicates default
        is_defaulter = "Defaulter" if results[0] == 1 else "Not a Defaulter"
        # Render the results on the home page
        return render_template('home.html', results=is_defaulter)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port = 8080)