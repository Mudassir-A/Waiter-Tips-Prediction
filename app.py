import pickle

import numpy as np
import pandas as pd

from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# importing model
model_regressor = pickle.load(open('models/linreg.pkl', 'rb'))

# route for home page
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/prediction", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        total_bill = float(request.form.get("total_bill"))
        sex = float(request.form.get("sex"))
        smoker = float(request.form.get("smoker"))
        day = float(request.form.get("day"))
        time = float(request.form.get("time"))
        size = float(request.form.get("size"))
        
        result = model_regressor.predict([[total_bill, sex, smoker, day, time, size]])[0]
        
        return render_template("home.html", result=round(result, 2))
    
    else: return render_template("home.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0")