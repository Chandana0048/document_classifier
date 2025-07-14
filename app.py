from flask import Flask, render_template, request, redirect, url_for
import os
from text_extractor import extract_text_from_pdf, extract_text_from_docx, extract_text_from_txt
import joblib
import pandas as pd
import csv
from datetime import datetime


# Initialize Flask App
app = Flask(__name__)

# Set Upload Folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained classifier model
model = joblib.load("models/document_classifier.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'document' not in request.files:
        return redirect(request.url)

    file = request.files['document']

    if file.filename == '':
        return redirect(request.url)

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Extract text based on file extension
        ext = file.filename.rsplit('.', 1)[1].lower()
        if ext == 'pdf':
            text = extract_text_from_pdf(file_path)
        elif ext == 'docx':
            text = extract_text_from_docx(file_path)
        elif ext == 'txt':
            text = extract_text_from_txt(file_path)
        else:
            text = "Unsupported file format."

        # Predict category
        prediction = model.predict([text])[0]
        probability = model.predict_proba([text]).max() * 100

        with open("classification_log.csv", "a", newline='', encoding='utf-8') as log_file:
            writer = csv.writer(log_file)
            writer.writerow([file.filename, prediction, round(probability, 2), datetime.now()])

        result = {
            'filename': file.filename,
            'prediction': prediction,
            'confidence': round(probability, 2),
            'extracted_text': text[:1000]  # limiting preview to first 1000 chars
        }

        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
