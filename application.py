import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask,jsonify,render_template,request

application=Flask(__name__)

app=application
#import ridge and standardScaler from models
ridge_model = pickle.load(open('models/ridge.pkl','rb'))
standard_scaler=pickle.load(open('models/scaler.pkl','rb'))



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        RH=float(request.form.get('RH'))
        WS=float(request.form.get('WS'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Classes=float(request.form.get('Classes'))

        data_scaled=standard_scaler.transform([[RH,WS,Rain,FFMC,DMC,ISI,Classes]])
        result=ridge_model.predict(data_scaled)

        return render_template('home.html',results=result)
        
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0')