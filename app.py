import os
from joblib import load
from flask import Flask, request, jsonify
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = SCRIPT_DIR

app = Flask(__name__)
# load model and vectorizer
model = load(os.path.join(PROJECT_DIR, 'savedModels/logistic_reg.joblib'))
vectorizer = load(os.path.join(PROJECT_DIR, 'data/tfidf/tfidf_vectorizer.joblib'))

@app.route('/predict', methods=['POST'])
def predict():

    data = request.json

    review_text = data['review']

    X = vectorizer.transform([review_text])


    prediction = model.predict(X)

    probab = model.predict_proba(X).max()

    sentiment_dict = {0: 'Negative', 1:'Neutral', 2:'Positive'}
    sentiment = sentiment_dict[prediction[0]]

    response = {
        'sentiment': sentiment,
        'confidence': float(probab)
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)