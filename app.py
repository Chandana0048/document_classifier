from flask import Flask, render_template, request, redirect, url_for , send_from_directory
import os
from text_extractor import extract_text_from_pdf, extract_text_from_docx, extract_text_from_txt
import joblib
import pandas as pd

import os
import csv
from datetime import datetime


# Initialize Flask App
app = Flask(__name__, static_folder='static')

# Set Upload Folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained classifier model
model = joblib.load("models/document_classifier.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/history')
def history():
    try:
        df = pd.read_csv("classification_log.csv")
        if df.empty:
            table_html = "<p style='text-align:center;'>No predictions yet. Upload documents to classify! 📄</p>"
        else:
            df.dropna(how='all', inplace=True)
            df.drop_duplicates(subset=["Filename", "Prediction", "Confidence (%)"], keep='last', inplace=True)
            df.reset_index(drop=True, inplace=True)
            table_html = df.to_html(classes='data', header="true", index=False, border=1)
    except FileNotFoundError:
        table_html = "<p style='text-align:center;'>No predictions yet. Upload documents to classify! 📄</p>"

    return render_template('history.html', table_html=table_html)




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
        
        log_file_path = "classification_log.csv"
        log_file_exists = os.path.exists("classification_log.csv")

        with open("classification_log.csv", "a", newline='', encoding='utf-8') as log_file:
           writer = csv.writer(log_file)
           if not log_file_exists:
              writer.writerow(["Filename", "Prediction", "Confidence (%)", "Timestamp"])
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
