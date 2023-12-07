from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
import numpy as np
import os

app = Flask(__name__)



base_dir = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(base_dir, 'models/sentiment_lstm_model.h5')
tokenizer_path = os.path.join(base_dir, 'models/tokenizer.json')
# Load trained model
model = load_model(model_path)  # Ensure this path is correct

# Load the tokenizer
with open(tokenizer_path, 'r') as file:
    tokenizer_data = json.load(file)
    tokenizer = tokenizer_from_json(tokenizer_data)

# Set max sequence length (should be the same as used during training)
max_seq_length = 121  # Update this to match the length used during training

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        sequence = tokenizer.texts_to_sequences([message])
        padded_sequence = pad_sequences(sequence, maxlen=max_seq_length)
        prediction = model.predict(padded_sequence)
        labels = ['Positive', 'Negative', 'Neutral']  # Update labels if needed
        sentiment = labels[np.argmax(prediction)]

        return render_template('index.html', prediction_text=f'Sentiment: {sentiment}')

if __name__ == '__main__':
    app.run(debug=True)
