from flask import Flask, jsonify, request,render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)   
app=application

ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))
@app.route('/')
def index():
    return render_template('home.html')      

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'POST':
        data = {
            "Temperature": float(request.form.get('Temperature')),
            "RH": float(request.form.get('Relative_Humidity')),
            "Ws": float(request.form.get('Wind_Speed')),
            "Rain": float(request.form.get('Rainfall')),
            "FFMC": float(request.form.get('FFMC')),
            "DMC": float(request.form.get('DMC')),
            "ISI": float(request.form.get('ISI')),
            "Classes": float(request.form.get('Classes')),
            "Region": float(request.form.get('Region'))
        }
        input_df = pd.DataFrame([data])
        scaled_data = standard_scaler.transform(input_df)
        prediction = ridge_model.predict(scaled_data)[0]
        return render_template('home.html', predictions=prediction)
    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
        
        