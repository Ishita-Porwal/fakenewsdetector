# Import necessary libraries
from flask import Flask, render_template, request, jsonify
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Initialize Flask app
app = Flask(__name__)

# Load and prepare the dataset (you can adjust this to load a saved model if you already trained it)
fake = pd.read_csv('Fake.csv')
true = pd.read_csv('True.csv')

# Labeling the data
fake['label'] = 0
true['label'] = 1

# Concatenating the data
news = pd.concat([fake, true], axis=0)

# Shuffling the data
news = news.sample(frac=1).reset_index(drop=True)

# Data cleaning function
def wordopt(text):
    text = text.lower()
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\n', ' ', text)
    return text

# Apply the cleaning function to the dataset
news['text'] = news['text'].apply(wordopt)

# Feature and label split
x = news['text']
y = news['label']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Vectorization
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# Train models
LR = LogisticRegression()
LR.fit(xv_train, y_train)

rfc = RandomForestClassifier()
rfc.fit(xv_train, y_train)

gbc = GradientBoostingClassifier()
gbc.fit(xv_train, y_train)

# Function to output label based on prediction
def output_label(n):
    return "Fake News" if n == 0 else "True News"

# Manual testing function
def manual_testing(news):
    news_cleaned = wordopt(news)
    new_xv_test = vectorization.transform([news_cleaned])

    pred_lr = LR.predict(new_xv_test)[0]
    pred_rfc = rfc.predict(new_xv_test)[0]
    pred_gbc = gbc.predict(new_xv_test)[0]

    return {
        'LR Prediction': output_label(pred_lr),
        'RFC Prediction': output_label(pred_rfc),
        'GBC Prediction': output_label(pred_gbc)
    }

# Define Flask routes
@app.route('/')
def home():
    return render_template('fakenewsdetector.html')  # Create an fakenewsdector.html file in templates folder


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_article = request.form['news_article']
        predictions = manual_testing(news_article)
        return jsonify(predictions)
    
    print("Flask app is starting...")
fake = pd.read_csv('Fake.csv')
print("Fake news data loaded")
true = pd.read_csv('True.csv')
print("True news data loaded")


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
