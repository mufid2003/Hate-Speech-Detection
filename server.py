from flask import Flask, render_template, request
import joblib
import re

app = Flask(__name__)

# load the hate speech detection model from the joblib file
model = joblib.load('hate_speech_model.joblib')


# define a function to preprocess the input text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove usernames
    text = re.sub(r'@\S+', '', text)
    # Remove hashtags
    text = re.sub(r'#\S+', '', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text


# define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')


# define the route for the prediction result
@app.route('/predict', methods=['POST'])
def predict():
    # get the input text from the HTML form
    text = request.form['text']

    # preprocess the input text
    preprocessed_text = preprocess_text(text)

    # predict whether the input text contains hate speech
    prediction = model.predict([preprocessed_text])[0]

    # format the prediction as a string
    if prediction == 0:
        result = 'Not hate speech'
    else:
        result = 'Hate speech'

    # render the prediction result on the HTML page
    return render_template('index.html', prediction_result=result)


if __name__ == '__main__':
    app.run(debug=True)
