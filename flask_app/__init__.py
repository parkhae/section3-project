from flask import Flask, render_template, request
import pandas as pd 
import pickle

app = Flask(__name__)

@app.route('/')
def main():
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict():
    gender = request.form['성별']
    age = float(request.form['나이'])
    hypertension = int(request.form['고혈압'])
    heart_disease = int(request.form['심장질환'])
    bmi = float(request.form['bmi'])
    smoking_status = request.form['흡연']
    
    model = None
    with open('model.pkl', 'rb') as pickle_file:
        model = pickle.load(pickle_file)
    
    X_data = pd.DataFrame([[gender,age,hypertension,heart_disease,bmi,smoking_status]], 
                        columns=['gender','age','hypertension','heart_disease','bmi','smoking_status'])
    
    #뇌졸중 위험 여부 예측
    pred = model.predict(X_data)
    return render_template('predict.html',data=pred)

if __name__ == '__main__':
    app.run(debug=True)
