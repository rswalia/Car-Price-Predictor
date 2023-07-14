from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
# import threading

app = Flask(__name__)
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
car = pd.read_csv('Cleaned Car.csv')


@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()

    companies.insert(0, 'Select Company')
    return render_template('index.html', companies=companies, car_models=car_models, years=year, fuel_types=fuel_type)


@app.route('/predict', methods=['POST'])
# @flask_cors.cross_origin()
def predict():
    company = request.form.get('company')

    name = request.form.get('car_models')
    year = request.form.get('year')
    fuel_type = request.form.get('fuel_type')
    kms_driven = int(request.form.get('kilo_driven'))

    print(company, name, year, fuel_type, kms_driven)
    data = {
        'name': [name],
        'company': [company],
        'year': [int(year)],
        'kms_driven': [kms_driven],
        'fuel_type': [fuel_type]
    }

    input_df = pd.DataFrame(data)
    prediction = model.predict(input_df)

    # prediction = model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']))
    print(str(prediction[0]))
    return str(prediction[0])


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)

# if __name__ == "__main__":
#     threading.Thread(target=app.run, kwargs={"debug": True, "use_reloader": False}).start()
