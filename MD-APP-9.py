# Project-9-lets Deploy False Alarm Detection Model End_to_End. -
import numpy as np
from flask import Flask, request, render_template
import joblib

APP9 = Flask(__name__)
pkl_file = joblib.load('train_md9.pkl')

@APP9.route('/')
def home():
    return render_template('home.html')

@APP9.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # For Rendering results on HTML GUI -
        f1 = float(request.form["Ambient Temperature( deg C)"])
        f2 = float(request.form["Calibration(days)"])
        f3 = float(request.form["Unwanted substance deposition(0/1)"])
        f4 = float(request.form["Humidity(%)"])
        f5 = float(request.form["H2S Content(ppm)"])
        f6 = float(request.form["detected by(% of sensors)"])

        data = np.array([[f1, f2, f3, f4, f5, f6]])
        predict_svmc = pkl_file.predict(data)
        return render_template('result.html', prediction=predict_svmc)

if __name__ == '__main__':
    APP9.run(debug=True)

'''
#Actual Inputs to check Output -
#False Alarm =1-Sample - {"Ambient Temperature":-2,"Calibration":226,"Unwanted substance deposition":1,"Humidity":96,"H2S Content":9,"detected by":21}
#True Alarm =0-Sample - {"Ambient Temperature":4,"Calibration":134.00,"Unwanted substance deposition":1,"Humidity":83,"H2S Content":4,"detected by":77}
'''
