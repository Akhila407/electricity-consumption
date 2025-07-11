from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        sqft = float(request.form['sqft'])
        occupants = int(request.form['occupants'])
        appliances = int(request.form['appliances'])
        
        # Predict consumption
        features = np.array([[sqft, occupants, appliances]])
        consumption = model.predict(features)[0]
        
        return redirect(url_for('result', consumption=consumption))
    
    return render_template('form.html')

@app.route('/result')
def result():
    consumption = float(request.args.get('consumption', 0))
    return render_template('result.html', consumption=consumption)

if __name__ == '__main__':
    app.run(debug=True)