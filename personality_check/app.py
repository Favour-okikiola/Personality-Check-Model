import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, render_template

# loading the train pickle file
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
print('model is successfully opened.')   
    
# creating flask app
app = Flask(__name__)

#app route
@app.route("/")
def Home():
    return render_template("index.html")

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {
            'Time_spent_Alone': float(request.form['Time_spent_Alone']),
            'Stage_fear': request.form['Stage_fear'],
            'Social_event_attendance': float(request.form['Social_event_attendance']),
            'Going_outside': float(request.form['Going_outside']),
            'Drained_after_socializing': request.form['Drained_after_socializing'],
            'Friends_circle_size': float(request.form['Friends_circle_size']),
            'Post_frequency': float(request.form['Post_frequency'])
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Make prediction
        prediction = model.predict(input_df)[0]
        output = "you are an introvert" if prediction == 'Personality'  else "You are an extrovert"

        return render_template('index.html', prediction=output)
    
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")
        
    # Run the app
if __name__ == '__main__':
    app.run(debug=True)