import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics
import joblib
import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')


# Sample dataset (or replace this with your own labeled doc dataset)
data = {
    'text': [
        'Payment for invoice #123',
        'Your application for software engineering role',
        'Terms and conditions for legal service',
        'Resume of John Doe',
        'Contract for property rental agreement',
        'Invoice payment pending',
        'Job offer for data analyst position'
    ],
    'category': [
        'Invoice',
        'Resume',
        'Legal',
        'Resume',
        'Legal',
        'Invoice',
        'Resume'
    ]
}
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    text = ' '.join(word for word in text.split() if word.lower() not in stop_words)
    return text.lower()


df = pd.DataFrame(data)
df['clean_text'] = df['text'].apply(clean_text)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['category'], test_size=0.2, random_state=42)

# Create a pipeline with Tfidf vectorizer + Naive Bayes classifier
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=200))
])

# Fit model
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
print(metrics.classification_report(y_test, predictions))

# Save model
joblib.dump(model, 'models/document_classifier.pkl')
print("✅ Model trained and saved as 'document_classifier.pkl'")
