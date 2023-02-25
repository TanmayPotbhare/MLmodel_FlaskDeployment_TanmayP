import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
sdf = pickle.load(open('StrokeP.pkl','rb'))


@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/predict', methods =['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = sdf.predict(final_features)
    
    output = round(prediction[0],2)
    
    return render_template('homepage.html',prediction_text = 'Stroke Prediction for this Patient is {}'.format(output))
        
if __name__ == '__main__':
    app.run(debug=True)
